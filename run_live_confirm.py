"""Run the live CORE-RT A/C confirmation with the existing LLM harness.

This script does not store credentials. Set GEMINI_API_KEY or OPENAI_API_KEY in the process
environment before running it.

Example:
    python run_live_confirm.py --data data/cases.jsonl --families A,C --out-prefix data/confirm_ac_live
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import binomtest


def load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" not in row:
            rows[row["id"]] = row
    return rows


def prepare_data(args: argparse.Namespace, prefix: Path) -> Path:
    source = Path(args.data)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    if not families:
        return source

    wanted = set(families)
    counts: Counter[str] = Counter()
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        family = row.get("family")
        if family not in wanted:
            continue
        if args.limit_per_family and counts[family] >= args.limit_per_family:
            continue
        rows.append(row)
        counts[family] += 1

    out = prefix.parent / f"{prefix.name}_input.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return out


def run_system(args: argparse.Namespace, data_path: Path, system: str, out_path: Path, log_path: Path) -> int:
    cmd = [
        sys.executable,
        "run.py",
        "--system",
        system,
        "--data",
        str(data_path),
        "--workers",
        str(args.workers),
        "--chunk",
        str(args.chunk),
        "--out",
        str(out_path),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def summarize(memory_path: Path, mapreduce_path: Path) -> dict:
    systems = {"memory": load_jsonl(memory_path), "mapreduce": load_jsonl(mapreduce_path)}
    ids = sorted(set(systems["memory"]) & set(systems["mapreduce"]))
    fam = defaultdict(lambda: Counter())
    for case_id in ids:
        family = systems["memory"][case_id]["family"]
        fam[family]["n"] += 1
        for name in systems:
            row = systems[name][case_id]
            fam[family][f"{name}_correct"] += int(row["correct"])
            fam[family][f"{name}_leaked"] += int(row["leaked"])
            fam[family][f"{name}_abstained"] += int(row["abstained"])

    by_family = {}
    for family, counts in sorted(fam.items()):
        n = counts["n"]
        by_family[family] = {"n": n}
        for name in systems:
            by_family[family][f"{name}_accuracy"] = round(counts[f"{name}_correct"] / n, 4)
            by_family[family][f"{name}_leak"] = round(counts[f"{name}_leaked"] / n, 4)
            by_family[family][f"{name}_abstain"] = round(counts[f"{name}_abstained"] / n, 4)

    a_ids = [i for i in ids if systems["memory"][i]["family"] == "A"]
    b = sum(1 for i in a_ids if systems["memory"][i]["correct"] and not systems["mapreduce"][i]["correct"])
    c = sum(1 for i in a_ids if not systems["memory"][i]["correct"] and systems["mapreduce"][i]["correct"])
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
    return {
        "paired_ids": len(ids),
        "by_family": by_family,
        "family_a_memory_only_right": b,
        "family_a_mapreduce_only_right": c,
        "family_a_mcnemar_p": round(p, 6),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/confirm_ac.jsonl")
    p.add_argument("--families", default="")
    p.add_argument("--limit-per-family", type=int, default=0)
    p.add_argument("--out-prefix", default="data/confirm_ac_live")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--chunk", type=int, default=12)
    args = p.parse_args()

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        raise SystemExit("Set GEMINI_API_KEY or OPENAI_API_KEY in the environment.")

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    data_path = prepare_data(args, prefix)
    map_path = prefix.parent / "res_mapreduce_confirm_ac_live.jsonl"
    mem_path = prefix.parent / "res_memory_confirm_ac_live.jsonl"
    summary_path = prefix.parent / "confirm_ac_live_summary.json"

    rc_map = run_system(args, data_path, "mapreduce", map_path, prefix.parent / "confirm_ac_live_mapreduce.log")
    rc_mem = run_system(args, data_path, "memory", mem_path, prefix.parent / "confirm_ac_live_memory.log")
    payload = {
        "kind": "live_llm_confirmation",
        "source_data": args.data,
        "run_data": str(data_path),
        "families": [f.strip() for f in args.families.split(",") if f.strip()],
        "limit_per_family": args.limit_per_family,
        "workers": args.workers,
        "chunk": args.chunk,
        "return_codes": {"mapreduce": rc_map, "memory": rc_mem},
        "summary": summarize(mem_path, map_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if rc_map or rc_mem:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
