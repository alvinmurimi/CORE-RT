"""Batched live CORE-RT confirmation.

This runner reduces API calls by grouping independent cases into one request. It keeps the comparison
paired: every case receives both a read-time resolver answer and a write-time extractor+resolver answer.

Credentials are read only from the process environment.

Example:
    python run_batched_live_confirm.py --data data/cases.jsonl --families A,C --limit-per-family 60 \
      --reader-batch-size 4 --memory-batch-size 2 --sleep 1.5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import binomtest

from run import ABSTAIN, Backend, _fuzzy, _json_array, resolve_cell
from score import score_answer


READER_SYS = (
    "Resolve independent CORE-RT cases. Some notes correct or retract earlier values. "
    "Rules for each case: if a note says an earlier value was an error, withdrawn, incorrect, "
    "mistaken, retracted, or a mix-up, that value is dead and cannot be the answer. The answer is the "
    "one stated value still alive. If two different values are both alive with nothing deciding between "
    "them, answer UNRESOLVED. If no value is retracted and one value is stated more than the others, "
    "that dominant value is the answer. Output ONLY a JSON array of objects with keys id and answer."
)

EXTRACT_BATCH_SYS = (
    "Extract records from independent CORE-RT cases. For a note that STATES a value "
    "(\"The <attribute> of <entity> is <value>\"), output an edge with subject, attribute, asserts, "
    "and retracts=null. For a note saying an earlier value for an attribute was an error, withdrawn, "
    "incorrect, mistaken, retracted, or a mix-up, output an edge with subject, attribute, asserts=null, "
    "and retracts equal to the value called wrong. Copy subject, attribute, and values verbatim. "
    "Output ONLY a JSON array of objects with keys id and edges. Each edges value is a JSON array."
)


def load_cases(args: argparse.Namespace) -> list[dict]:
    wanted = {f.strip() for f in args.families.split(",") if f.strip()}
    counts: Counter[str] = Counter()
    cases: list[dict] = []
    for line in Path(args.data).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        family = row.get("family")
        if wanted and family not in wanted:
            continue
        if args.limit_per_family and counts[family] >= args.limit_per_family:
            continue
        cases.append(row)
        counts[family] += 1
    return cases


def chunks(xs: list[dict], n: int) -> list[list[dict]]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def case_block(case: dict) -> str:
    notes = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(case["memories"]))
    return (
        f"CASE {case['id']}\n"
        f"Question: {case['question']}\n"
        f"Notes:\n{notes}"
    )


def parse_array_by_id(text: str) -> dict[str, dict]:
    rows = _json_array(text)
    out = {}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("id"):
            out[str(row["id"])] = row
    return out


def reader_prompt(batch: list[dict]) -> str:
    return (
        "Resolve each case independently. Return JSON only.\n\n"
        + "\n\n".join(case_block(case) for case in batch)
        + "\n\nJSON array schema: [{\"id\":\"case id\",\"answer\":\"value or UNRESOLVED\"}]"
    )


def extractor_prompt(batch: list[dict]) -> str:
    return (
        "Extract edges for each case independently. Return JSON only.\n\n"
        + "\n\n".join(case_block(case) for case in batch)
        + "\n\nJSON array schema: "
        "[{\"id\":\"case id\",\"edges\":[{\"subject\":\"...\",\"attribute\":\"...\","
        "\"asserts\":\"value or null\",\"retracts\":\"value or null\"}]}]"
    )


def batch_reader(backend: Backend, batch: list[dict], sleep_s: float) -> list[dict]:
    try:
        rows = parse_array_by_id(backend.chat(READER_SYS, reader_prompt(batch)))
        missing = [case for case in batch if case["id"] not in rows]
        if missing:
            raise RuntimeError(f"missing ids: {[c['id'] for c in missing]}")
        out = []
        for case in batch:
            answer = str(rows[case["id"]].get("answer", ""))
            out.append({"case": case, "answer": answer, "error": None})
        time.sleep(sleep_s)
        return out
    except Exception as exc:
        if len(batch) == 1:
            return [{"case": batch[0], "answer": "", "error": str(exc)[:300]}]
        mid = len(batch) // 2
        return batch_reader(backend, batch[:mid], sleep_s) + batch_reader(backend, batch[mid:], sleep_s)


def batch_extract(backend: Backend, batch: list[dict], sleep_s: float) -> list[dict]:
    try:
        rows = parse_array_by_id(backend.chat(EXTRACT_BATCH_SYS, extractor_prompt(batch)))
        missing = [case for case in batch if case["id"] not in rows]
        if missing:
            raise RuntimeError(f"missing ids: {[c['id'] for c in missing]}")
        out = []
        for case in batch:
            edges = rows[case["id"]].get("edges", [])
            if not isinstance(edges, list):
                raise RuntimeError(f"bad edges for {case['id']}")
            out.append({"case": case, "edges": edges, "error": None})
        time.sleep(sleep_s)
        return out
    except Exception as exc:
        if len(batch) == 1:
            return [{"case": batch[0], "edges": [], "error": str(exc)[:300]}]
        mid = len(batch) // 2
        return batch_extract(backend, batch[:mid], sleep_s) + batch_extract(backend, batch[mid:], sleep_s)


def resolve_from_edges(case: dict, edges: list[dict]) -> str:
    asserted: list[str] = []
    killed: set[str] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        if _fuzzy(str(edge.get("subject", "")), case["entity"]) and _fuzzy(
            str(edge.get("attribute", "")), case["attribute"]
        ):
            if edge.get("asserts"):
                asserted.append(str(edge["asserts"]))
            if edge.get("retracts"):
                killed.add(str(edge["retracts"]))
    return resolve_cell(asserted, killed)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def summarize(memory_rows: list[dict], reader_rows: list[dict]) -> dict:
    memory = {r["id"]: r for r in memory_rows if "error" not in r}
    reader = {r["id"]: r for r in reader_rows if "error" not in r}
    paired = sorted(set(memory) & set(reader))
    fam = defaultdict(lambda: Counter())
    for case_id in paired:
        family = memory[case_id]["family"]
        fam[family]["n"] += 1
        for name, rows in (("memory", memory), ("reader", reader)):
            fam[family][f"{name}_correct"] += int(rows[case_id]["correct"])
            fam[family][f"{name}_leaked"] += int(rows[case_id]["leaked"])
            fam[family][f"{name}_abstained"] += int(rows[case_id]["abstained"])

    by_family = {}
    for family, counts in sorted(fam.items()):
        n = counts["n"]
        by_family[family] = {"n": n}
        for name in ("memory", "reader"):
            by_family[family][f"{name}_accuracy"] = round(counts[f"{name}_correct"] / n, 4)
            by_family[family][f"{name}_leak"] = round(counts[f"{name}_leaked"] / n, 4)
            by_family[family][f"{name}_abstain"] = round(counts[f"{name}_abstained"] / n, 4)

    a_ids = [case_id for case_id in paired if memory[case_id]["family"] == "A"]
    b = sum(1 for case_id in a_ids if memory[case_id]["correct"] and not reader[case_id]["correct"])
    c = sum(1 for case_id in a_ids if not memory[case_id]["correct"] and reader[case_id]["correct"])
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
    return {
        "paired_ids": len(paired),
        "by_family": by_family,
        "family_a_memory_only_right": b,
        "family_a_reader_only_right": c,
        "family_a_mcnemar_p": round(p, 6),
        "memory_errors": len(memory_rows) - len(memory),
        "reader_errors": len(reader_rows) - len(reader),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/cases.jsonl")
    p.add_argument("--families", default="A,C")
    p.add_argument("--limit-per-family", type=int, default=20)
    p.add_argument("--out-prefix", default="data/confirm_ac_batched_live")
    p.add_argument("--reader-batch-size", type=int, default=4)
    p.add_argument("--memory-batch-size", type=int, default=2)
    p.add_argument("--sleep", type=float, default=1.5)
    p.add_argument("--base-url", default="https://generativelanguage.googleapis.com/v1beta/openai")
    p.add_argument("--chat-model", default="gemini-2.5-flash-lite")
    p.add_argument("--embed-model", default="gemini-embedding-001")
    p.add_argument("--dims", type=int, default=256)
    args = p.parse_args()

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY or OPENAI_API_KEY in the environment.")

    cases = load_cases(args)
    backend = Backend(args.base_url, key, args.chat_model, args.embed_model, args.dims)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    reader_rows = []
    for batch in chunks(cases, args.reader_batch_size):
        for item in batch_reader(backend, batch, args.sleep):
            case = item["case"]
            if item["error"]:
                reader_rows.append({"id": case["id"], "error": item["error"]})
                continue
            sc = score_answer(case, item["answer"])
            reader_rows.append({"id": case["id"], "family": case["family"], "answer": item["answer"], **sc})
        write_jsonl(prefix.parent / f"{prefix.name}_reader.jsonl", reader_rows)
        print(f"reader {len(reader_rows)}/{len(cases)}", flush=True)

    memory_rows = []
    for batch in chunks(cases, args.memory_batch_size):
        for item in batch_extract(backend, batch, args.sleep):
            case = item["case"]
            if item["error"]:
                memory_rows.append({"id": case["id"], "error": item["error"]})
                continue
            answer = resolve_from_edges(case, item["edges"])
            sc = score_answer(case, answer)
            memory_rows.append({
                "id": case["id"],
                "family": case["family"],
                "answer": answer,
                "edge_count": len(item["edges"]),
                **sc,
            })
        write_jsonl(prefix.parent / f"{prefix.name}_memory.jsonl", memory_rows)
        print(f"memory {len(memory_rows)}/{len(cases)}", flush=True)

    payload = {
        "kind": "live_llm_batched_confirmation",
        "source_data": args.data,
        "families": [f.strip() for f in args.families.split(",") if f.strip()],
        "limit_per_family": args.limit_per_family,
        "reader_batch_size": args.reader_batch_size,
        "memory_batch_size": args.memory_batch_size,
        "chat_model": args.chat_model,
        "summary": summarize(memory_rows, reader_rows),
    }
    (prefix.parent / f"{prefix.name}_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
