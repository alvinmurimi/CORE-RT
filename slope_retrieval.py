"""CORE-RT long-context slope, retrieval axis, isolated and LLM-free.

The question: as the number of notes N grows, does memory keep the answer while a read-time reader
loses it? This script isolates the RETRIEVAL half of that question from the extraction half (which
Section 7 already settled: memory pays an extraction tax read-time does not).

Method: hold extraction PERFECT (use the hidden provenance edges, no LLM), and vary only which notes a
policy surfaces at the fixed reader budget k, across N. Then apply the SAME deterministic resolution
rule the memory and map-reduce systems share. Score with the project's deterministic scorer.

Three policies, all reading at the same budget:
  sem_topk    : pure cosine top-k over the LSA retriever. The naive RAG baseline.
  cell_scoped : notes matching the queried (entity, attribute) cell. This is what memory routing does,
                and what a fair read-time reader does with a slot-keyed lane. Bounded
                and independent of N.
  hybrid      : top-(k-m) semantic plus up to m cell-scoped, the realistic read-time hybrid.

If sem_topk degrades with N while cell_scoped and hybrid stay flat, the slope effect is REAL but it is a
retrieval-lane property available to a read-time reader, NOT a write-time-memory advantage: memory ties
the fair reader. That is the guardrail the critic asked for (no weak-baseline false positive).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from generate import build_case
from local_embed import LocalEmbedder
from score import score_answer

ABSTAIN = "UNRESOLVED"


def _norm(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", re.sub(r"^the ", "", (s or "").strip().lower())).strip(" .")


def resolve_cell(asserted: list[str], killed: set[str]) -> str:
    """Identical to run.resolve_cell: kills win; lone survivor or strict majority answers; else abstain."""
    killed_n = {_norm(k) for k in killed}
    alive = Counter(v for v in asserted if _norm(v) not in killed_n)
    if not alive:
        return ABSTAIN
    if len(alive) == 1:
        return next(iter(alive))
    (v1, c1), (v2, c2) = alive.most_common(2)
    return v1 if c1 > c2 else ABSTAIN


def cell_resolve_from_indices(case: dict, idxs) -> tuple[str, bool]:
    """Apply perfect extraction over the SURFACED notes: keep only provenance edges on the queried
    (entity, attribute) cell, then resolve. Returns (answer, retraction_was_surfaced)."""
    ent, attr = case["entity"], case["attribute"]
    asserted: list[str] = []
    killed: set[str] = set()
    saw_retraction = False
    for i in idxs:
        p = case["provenance"][i]
        if p.get("subj") != ent or p.get("attr") != attr:
            continue
        if p.get("role") == "retraction" and p.get("kills"):
            killed.add(p["kills"])
            saw_retraction = True
        elif p.get("value"):
            asserted.append(p["value"])
    return resolve_cell(asserted, killed), saw_retraction


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def sem_order(case, embedder) -> list[int]:
    """Full semantic ranking of every note by cosine to the query (fit once, reused across k)."""
    mems = case["memories"]
    q = embedder.embed([case["question"]])[0]
    vecs = embedder.embed(mems)
    return sorted(range(len(mems)), key=lambda i: cosine(vecs[i], q), reverse=True)


def cell_indices(case) -> list[int]:
    ent, attr = case["entity"], case["attribute"]
    return [i for i, p in enumerate(case["provenance"])
            if p.get("subj") == ent and p.get("attr") == attr]


POLICIES = ("sem_topk", "cell_scoped", "hybrid")


def eval_policy(case, policy, order, cells, k, m) -> dict:
    if policy == "sem_topk":
        idxs = order[:k]
    elif policy == "cell_scoped":
        idxs = cells[:max(k, 8)]  # the cell is small; never starve it below the cell size
    elif policy == "hybrid":
        cm = cells[:m]
        sem = [i for i in order if i not in set(cm)]
        idxs = (cm + sem)[:k]
    else:
        raise ValueError(policy)
    ans, saw_retr = cell_resolve_from_indices(case, idxs)
    sc = score_answer(case, ans)
    return {"correct": sc["correct"], "leaked": sc["leaked"], "saw_retraction": saw_retr}


def run_sweep(per_family, k_poison, Ns, ks, seed, hybrid_m, dims):
    rows = []
    for N in Ns:
        # Family A only: the explicit-correction core where the slope could bite.
        cases = [build_case("A", i, k_poison, N, seed, negation_only=(i % 2 == 0))
                 for i in range(per_family)]
        # Fit the LSA retriever ONCE per case and cache the full semantic ranking; k only slices it.
        prepared = []
        for c in cases:
            emb = LocalEmbedder(c["memories"] + [c["question"]], dims=dims, seed=seed)
            prepared.append((c, sem_order(c, emb), cell_indices(c)))
        for k in ks:
            agg = {pol: {"n": 0, "correct": 0, "leaked": 0, "saw_retr": 0} for pol in POLICIES}
            for c, order, cells in prepared:
                for pol in POLICIES:
                    r = eval_policy(c, pol, order, cells, k, hybrid_m)
                    a = agg[pol]
                    a["n"] += 1
                    a["correct"] += int(r["correct"])
                    a["leaked"] += int(r["leaked"])
                    a["saw_retr"] += int(r["saw_retraction"])
            for pol in POLICIES:
                a = agg[pol]
                n = a["n"]
                rows.append({"N": N, "k": k, "policy": pol, "n": n,
                             "accuracy": round(a["correct"] / n, 3),
                             "leak": round(a["leaked"] / n, 3),
                             "retraction_recall": round(a["saw_retr"] / n, 3)})
            print(f"N={N:5d} k={k:3d} | "
                  + "  ".join(f"{pol}: acc={[r for r in rows if r['N']==N and r['k']==k and r['policy']==pol][0]['accuracy']:.2f} "
                              f"leak={[r for r in rows if r['N']==N and r['k']==k and r['policy']==pol][0]['leak']:.2f}"
                              for pol in POLICIES), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description="CORE-RT retrieval-axis long-context slope (LLM-free).")
    p.add_argument("--per-family", type=int, default=30)
    p.add_argument("--k-poison", type=int, default=4)
    p.add_argument("--Ns", default="30,120,480,1920,5000")
    p.add_argument("--ks", default="10,20,40")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--hybrid-m", type=int, default=6)
    p.add_argument("--dims", type=int, default=128)
    p.add_argument("--out", default="data/slope_retrieval.json")
    a = p.parse_args()
    Ns = [int(x) for x in a.Ns.split(",")]
    ks = [int(x) for x in a.ks.split(",")]
    rows = run_sweep(a.per_family, a.k_poison, Ns, ks, a.seed, a.hybrid_m, a.dims)
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"config": vars(a), "rows": rows}, indent=2), encoding="utf-8")
    print("\nWROTE", a.out)


if __name__ == "__main__":
    main()
