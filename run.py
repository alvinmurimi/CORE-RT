"""CORE-RT harness: score the four systems on the timestamp-free, provenance-gold benchmark.

Systems (see SPEC.md):
  plain     : LSA top-k over raw sentences, reader with NO correction instruction (floor).
  single    : whole case in one window, told to honor corrections + abstain (strawman calibration).
  mapreduce : LLM folds the FULL pile chunk-by-chunk into a running answer at READ time (decisive baseline).
  memory    : LLM extracts edges from prose at WRITE time, routes them per (subject, attribute) cell, and a
              DETERMINISTIC resolver applies the same rules. Reads return the cached cell.

map-reduce and memory share the SAME resolution rules; they differ only in resolve-each-query (read) vs
resolve-once-and-cache (write). Scoring is deterministic substring matching against the case's known
values (novel strings) -- no judge LLM. --closed-book empties the context to probe parametric leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from score import score_answer

ABSTAIN = "UNRESOLVED"


class Backend:
    def __init__(self, base_url, key, chat_model, embed_model, dims):
        self.c = httpx.Client(base_url=base_url.rstrip("/"), timeout=60)
        self.key, self.chat_model, self.embed_model, self.dims = key, chat_model, embed_model, dims

    def _post(self, path, payload):
        last, last_status = None, None
        for attempt in range(8):
            try:
                r = self.c.post(path, json=payload, headers={"Authorization": f"Bearer {self.key}"})
            except httpx.TransportError as exc:
                last = exc
                time.sleep(min(2**attempt, 30) + random.uniform(0, 0.75))
                continue
            if r.status_code in {429, 500, 502, 503, 504}:
                last_status = r.status_code
                time.sleep(min(2**attempt, 30) + random.uniform(0, 0.75))
                continue
            r.raise_for_status()
            return r.json()
        if last:
            raise last
        raise RuntimeError(f"{path} failed after retries (last HTTP {last_status})")

    def chat(self, system, user):
        d = self._post("/chat/completions", {"model": self.chat_model, "temperature": 0.0,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]})
        return str(d["choices"][0]["message"]["content"]).strip()


# ---------- shared deterministic resolution rule ----------

def resolve_cell(asserted: list[str], killed: set[str]) -> str:
    """Identical rule used by memory (write time) and as the final fold of map-reduce. Kills always win;
    among the values still alive, a single survivor is the answer, a strict majority is the answer (the
    no-correction regime), and a genuine tie or total wipeout is ABSTAIN."""
    killed_n = {_norm(k) for k in killed}
    alive = Counter(v for v in asserted if _norm(v) not in killed_n)
    if not alive:
        return ABSTAIN
    if len(alive) == 1:
        return next(iter(alive))
    (v1, c1), (v2, c2) = alive.most_common(2)
    return v1 if c1 > c2 else ABSTAIN


# ---------- LLM building blocks ----------

def _json_array(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text).strip()
    i, j = text.find("["), text.rfind("]")
    if i == -1 or j == -1:
        return []
    try:
        return json.loads(text[i:j + 1])
    except Exception:
        return []


EXTRACT_SYS = (
    "Extract one record per note. For a note that STATES a value (\"The <attribute> of <entity> is "
    "<value>\"), output {\"subject\":<entity copied exactly>,\"attribute\":<attribute copied exactly>,"
    "\"asserts\":<value copied exactly>,\"retracts\":null}. For a note saying an earlier value for an "
    "attribute was an error / withdrawn / incorrect / mistaken / retracted / a mix-up, output "
    "{\"subject\":<entity>,\"attribute\":<attribute>,\"asserts\":null,\"retracts\":<the value called "
    "wrong, copied exactly>}. Copy subject, attribute and value VERBATIM including any leading 'the'. "
    "Emit a record for EVERY note that states or retracts a value; skip only notes that do neither. "
    "Output ONLY a JSON array, one object per such note."
)

MAPREDUCE_SYS = (
    "You track the current {attr} of {entity} as notes arrive; some notes correct or retract earlier ones. "
    "Rules: if a note says an earlier value was an error/withdrawn/incorrect, that value is dead and cannot "
    "be the answer. The answer is the one stated value still alive. If two different values are both alive "
    "with nothing deciding between them, answer " + ABSTAIN + ". If no value is retracted and one value is "
    "stated more than the others, that dominant value is the answer. Output ONLY the current value, or "
    + ABSTAIN + "."
)

SINGLE_SYS = (
    "Answer using ALL the notes. Some notes correct or retract earlier ones; never use a value that was "
    "called an error/withdrawn/incorrect, use the corrected current one. If two values are equally "
    "supported with nothing deciding between them, reply " + ABSTAIN + ". Reply with ONLY the value or "
    + ABSTAIN + "."
)

PLAIN_SYS = "Answer the question from the notes. Reply with ONLY the value, in one short phrase."


def _chunks(xs: list, n: int) -> list[list]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def answer_plain(case, backend, embedder, k):
    mems = case["memories"]
    q = embedder.embed([case["question"]])[0]
    vecs = embedder.embed(mems)
    idx = sorted(range(len(mems)), key=lambda i: sum(a * b for a, b in zip(vecs[i], q)), reverse=True)[:k]
    notes = "\n".join(f"- {mems[i]}" for i in idx)
    return backend.chat(PLAIN_SYS, f"Notes:\n{notes}\n\nQuestion: {case['question']}")


def answer_single(case, backend):
    notes = "\n".join(f"- {m}" for m in case["memories"])
    return backend.chat(SINGLE_SYS, f"Notes:\n{notes}\n\nQuestion: {case['question']}")


def answer_mapreduce(case, backend, chunk):
    sys = MAPREDUCE_SYS.format(attr=case["attribute"], entity=case["entity"])
    state = f"{ABSTAIN} (no notes yet)"
    for ch in _chunks(case["memories"], chunk):
        block = "\n".join(f"- {m}" for m in ch)
        state = backend.chat(sys, f"Current answer: {state}\nNew notes:\n{block}\n\n"
                                  f"Updated current {case['attribute']} of {case['entity']}:")
    return state


def answer_memory(case, backend, chunk):
    """Write-time: extract edges per chunk, route to the queried cell by (subject, attribute), then a
    deterministic resolver applies the shared rules. Returns the cached cell value (or ABSTAIN).
    Routing is fuzzy (normalized containment) so a slightly paraphrased subject/attribute from the
    extractor still lands in the right cell rather than being silently dropped."""
    asserted: list[str] = []
    killed: set[str] = set()
    for ch in _chunks(case["memories"], chunk):
        block = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(ch))
        try:
            edges = _json_array(backend.chat(EXTRACT_SYS, f"Notes:\n{block}\n\nJSON:"))
        except Exception:
            edges = []
        for e in edges if isinstance(edges, list) else []:
            if not isinstance(e, dict):
                continue
            if not (_fuzzy(str(e.get("subject", "")), case["entity"])
                    and _fuzzy(str(e.get("attribute", "")), case["attribute"])):
                continue
            if e.get("asserts"):
                asserted.append(str(e["asserts"]))
            if e.get("retracts"):
                killed.add(str(e["retracts"]))
    return resolve_cell(asserted, killed)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"^the ", "", (s or "").strip().lower())).strip(" .")


def _fuzzy(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def answer_closedbook(case, backend):
    return backend.chat(SINGLE_SYS, f"Notes:\n(no notes)\n\nQuestion: {case['question']}")


SYSTEMS = {"plain", "single", "mapreduce", "memory"}


def run(system, backend, embedder, cases, out_path, workers, k, chunk, closed_book):
    results, errors, done = [], 0, 0
    fam_stat: dict[str, list[int]] = {}
    fh = Path(out_path).open("w", encoding="utf-8") if out_path else None
    lock = threading.Lock()
    total = len(cases)

    def work(case):
        nonlocal errors, done
        try:
            if closed_book:
                ans = answer_closedbook(case, backend)
            elif system == "plain":
                ans = answer_plain(case, backend, embedder, k)
            elif system == "single":
                ans = answer_single(case, backend)
            elif system == "mapreduce":
                ans = answer_mapreduce(case, backend, chunk)
            elif system == "memory":
                ans = answer_memory(case, backend, chunk)
            else:
                raise ValueError(system)
            sc = score_answer(case, ans)
        except Exception as exc:
            with lock:
                errors += 1
                if fh:
                    fh.write(json.dumps({"id": case["id"], "error": str(exc)[:200]}) + "\n")
                    fh.flush()
            return
        rec = {"id": case["id"], "family": case["family"], "negation_only": case.get("negation_only"),
               "answer": ans, **sc, "gold": case["gold"]}
        with lock:
            done += 1
            results.append(rec)
            st = fam_stat.setdefault(case["family"], [0, 0, 0, 0])  # n, correct, leaked, abstained
            st[0] += 1
            st[1] += sc["correct"]
            st[2] += sc["leaked"]
            st[3] += sc["abstained"]
            if fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            if done % 20 == 0 or done == total:
                acc = sum(s[1] for s in fam_stat.values()) / done
                print(f"  [{done}/{total}] acc={acc:.3f}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, cases))
    finally:
        if fh:
            fh.close()
    by_family = {f: {"n": s[0], "accuracy": round(s[1] / s[0], 4) if s[0] else 0,
                     "leak": round(s[2] / s[0], 4) if s[0] else 0,
                     "abstain": round(s[3] / s[0], 4) if s[0] else 0}
                 for f, s in sorted(fam_stat.items())}
    return {"system": system, "closed_book": closed_book, "n": len(results), "errors": errors,
            "by_family": by_family}


def main():
    p = argparse.ArgumentParser(description="Score CORE-RT systems.")
    p.add_argument("--system", default="memory", choices=sorted(SYSTEMS))
    p.add_argument("--data", default="data/cases.jsonl")
    p.add_argument("--base-url", default="https://generativelanguage.googleapis.com/v1beta/openai")
    p.add_argument("--chat-model", default="gemini-2.5-flash-lite")
    p.add_argument("--embed-model", default="gemini-embedding-001")
    p.add_argument("--embedder", default="local", choices=["gemini", "local"])
    p.add_argument("--dims", type=int, default=256)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--chunk", type=int, default=12)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--closed-book", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY in the environment.")
    cases = [json.loads(l) for l in Path(a.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        cases = cases[:a.limit]
    backend = Backend(a.base_url, key, a.chat_model, a.embed_model, a.dims)
    embedder = None
    if a.system == "plain" and not a.closed_book:
        from local_embed import LocalEmbedder
        corpus = [m for c in cases for m in c["memories"]] + [c["question"] for c in cases]
        print(f"fitting local LSA embedder on {len(corpus)} texts ...", flush=True)
        embedder = LocalEmbedder(corpus, dims=a.dims)
    summary = run(a.system, backend, embedder, cases, a.out, a.workers, a.k, a.chunk, a.closed_book)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
