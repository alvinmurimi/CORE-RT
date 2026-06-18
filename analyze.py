"""Aggregate CORE-RT result files and apply the kill criteria. Offline; no API.

Reads one results JSONL per system (the --out files from run.py), joins them by case id, and reports
per-family accuracy / poison-leak / abstention, the paired McNemar test (memory vs the read-time
map-reduce reader) on the non-tautological family A, and a literal check of the kill criteria.

    python analyze.py plain=results/plain.jsonl single=results/single.jsonl \
        mapreduce=results/mapreduce.jsonl memory=results/memory.jsonl closedbook=results/closedbook.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

from scipy.stats import binomtest


def load(path: str) -> dict[str, dict]:
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" not in r:
            out[r["id"]] = r
    return out


def main():
    systems = {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        systems[name] = load(path)
    if not systems:
        raise SystemExit("usage: analyze.py name=path.jsonl ...")

    # union of families present
    fams = sorted({r["family"] for s in systems.values() for r in s.values()})
    order = [s for s in ["plain", "single", "mapreduce", "memory", "closedbook"] if s in systems]
    order += [s for s in systems if s not in order]

    print("=" * 78)
    print("ACCURACY by family (the non-tautological core is A; B/D are guardrails; C is abstention)")
    print("=" * 78)
    print(f"{'family':<8}" + "".join(f"{s:>13}" for s in order))
    for fam in fams:
        row = f"{fam:<8}"
        for s in order:
            recs = [r for r in systems[s].values() if r["family"] == fam]
            acc = sum(r["correct"] for r in recs) / len(recs) if recs else float("nan")
            row += f"{acc:>13.3f}"
        print(row)

    print("\nPOISON-LEAK by family (lower better)")
    print(f"{'family':<8}" + "".join(f"{s:>13}" for s in order))
    for fam in fams:
        row = f"{fam:<8}"
        for s in order:
            recs = [r for r in systems[s].values() if r["family"] == fam]
            leak = sum(r["leaked"] for r in recs) / len(recs) if recs else float("nan")
            row += f"{leak:>13.3f}"
        print(row)

    # paired McNemar: memory vs mapreduce, family A
    if "memory" in systems and "mapreduce" in systems:
        print("\n" + "=" * 78)
        print("PAIRED McNEMAR  memory vs mapreduce  (family A)")
        print("=" * 78)
        ids = [i for i in systems["memory"] if i in systems["mapreduce"]
               and systems["memory"][i]["family"] == "A"]
        b = sum(1 for i in ids if systems["memory"][i]["correct"] and not systems["mapreduce"][i]["correct"])
        c = sum(1 for i in ids if not systems["memory"][i]["correct"] and systems["mapreduce"][i]["correct"])
        both = sum(1 for i in ids if systems["memory"][i]["correct"] and systems["mapreduce"][i]["correct"])
        neither = len(ids) - b - c - both
        n = b + c
        p = binomtest(min(b, c), n, 0.5).pvalue if n else 1.0
        mem_acc = sum(systems["memory"][i]["correct"] for i in ids) / len(ids) if ids else 0
        mr_acc = sum(systems["mapreduce"][i]["correct"] for i in ids) / len(ids) if ids else 0
        print(f"  family-A cases paired: {len(ids)}")
        print(f"  memory acc={mem_acc:.3f}   mapreduce acc={mr_acc:.3f}")
        print(f"  memory-only-right b={b}   mapreduce-only-right c={c}   both={both}   neither={neither}")
        print(f"  McNemar exact p={p:.4f}")
        verdict = ("memory SIGNIFICANTLY beats map-reduce on A" if p < 0.05 and b > c else
                   "map-reduce SIGNIFICANTLY beats memory on A" if p < 0.05 and c > b else
                   "NO significant difference on A (kill-criterion #1 triggered)")
        print(f"  -> {verdict}")

    # contamination check
    if "closedbook" in systems:
        print("\nCLOSED-BOOK accuracy by family (must be ~0 on novel strings, else parametric leak)")
        for fam in fams:
            recs = [r for r in systems["closedbook"].values() if r["family"] == fam]
            acc = sum(r["correct"] for r in recs) / len(recs) if recs else float("nan")
            flag = "  <-- LEAK" if (recs and acc > 0.15 and fam == "A") else ""
            print(f"  {fam}: {acc:.3f}{flag}")


if __name__ == "__main__":
    main()
