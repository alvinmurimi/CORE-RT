"""Full-size deterministic confirming sweep for CORE-RT.

This is not an LLM run. It is a visible-text, grammar-aware parser over the generated notes, followed
by the same resolver used by memory and read-time map-reduce. Hidden provenance is used only by the
scorer, never by the parser.

Use this when model quota is unavailable to check the benchmark contract at larger n:

    python confirm_deterministic.py --data data/cases.jsonl --out data/confirm_deterministic_full.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from run import _fuzzy, resolve_cell
from score import score_answer


PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(
        r"^An earlier note recording that the (?P<attr>.+?) of (?P<entity>.+?) is "
        r"(?P<retracts>.+?) was filed in error and withdrawn\.$"
    ), "retract"),
    (re.compile(
        r"^The claim that the (?P<attr>.+?) of (?P<entity>.+?) is (?P<retracts>.+?) "
        r"proved mistaken and should be disregarded\.$"
    ), "retract"),
    (re.compile(
        r"^Correction: (?P<entity>.+?) does not have (?P<retracts>.+?) as its "
        r"(?P<attr>.+?); that entry was a mix-up\.$"
    ), "retract"),
    (re.compile(
        r"^It has since been established that naming (?P<retracts>.+?) as the "
        r"(?P<attr>.+?) of (?P<entity>.+?) was incorrect\.$"
    ), "retract"),
    (re.compile(
        r"^The record listing (?P<retracts>.+?) as the (?P<attr>.+?) of "
        r"(?P<entity>.+?) has been retracted as erroneous\.$"
    ), "retract"),
    (re.compile(
        r"^Disregard the report that the (?P<attr>.+?) of (?P<entity>.+?) is "
        r"(?P<retracts>.+?); it was logged by mistake\.$"
    ), "retract"),
    (re.compile(
        r"^An earlier note naming (?P<retracts>.+?) as the (?P<attr>.+?) of (?P<entity>.+?) "
        r"was an error; the (?P=attr) is in fact (?P<asserts>.+?)\.$"
    ), "retract_assert"),
    (re.compile(
        r"^Correction: the (?P<attr>.+?) of (?P<entity>.+?) was wrongly given as "
        r"(?P<retracts>.+?); it is actually (?P<asserts>.+?)\.$"
    ), "retract_assert"),
    (re.compile(
        r"^The entry stating the (?P<attr>.+?) of (?P<entity>.+?) is (?P<retracts>.+?) "
        r"was mistaken; the correct value is (?P<asserts>.+?)\.$"
    ), "retract_assert"),
    (re.compile(
        r"^(?P<retracts>.+?) was recorded in error as the (?P<attr>.+?) of (?P<entity>.+?); "
        r"that role belongs to (?P<asserts>.+?)\.$"
    ), "retract_assert"),
    (re.compile(r"^The (?P<attr>.+?) of (?P<entity>.+?) is (?P<asserts>.+?)\.$"), "assert"),
]


def parse_note(note: str) -> dict | None:
    for pat, kind in PATTERNS:
        m = pat.match(note.strip())
        if not m:
            continue
        d = m.groupdict()
        return {
            "subject": d.get("entity"),
            "attribute": d.get("attr"),
            "asserts": d.get("asserts") if kind in {"assert", "retract_assert"} else None,
            "retracts": d.get("retracts") if kind in {"retract", "retract_assert"} else None,
        }
    return None


def resolve_from_visible_notes(case: dict, notes: list[str]) -> tuple[str, list[dict]]:
    edges = [e for note in notes if (e := parse_note(note))]
    asserted: list[str] = []
    killed: set[str] = set()
    for e in edges:
        if _fuzzy(str(e.get("subject", "")), case["entity"]) and _fuzzy(str(e.get("attribute", "")), case["attribute"]):
            if e.get("asserts"):
                asserted.append(str(e["asserts"]))
            if e.get("retracts"):
                killed.add(str(e["retracts"]))
    return resolve_cell(asserted, killed), edges


def run_case(case: dict, chunk: int) -> dict:
    memory_answer, edges = resolve_from_visible_notes(case, case["memories"])

    # Read-time map-reduce equivalent: parse surfaced chunks in order, but keep the same cumulative cell
    # state and deterministic final resolver. This differs from memory only in when the same work occurs.
    asserted: list[str] = []
    killed: set[str] = set()
    for i in range(0, len(case["memories"]), chunk):
        _, chunk_edges = resolve_from_visible_notes(case, case["memories"][i:i + chunk])
        for e in chunk_edges:
            if _fuzzy(str(e.get("subject", "")), case["entity"]) and _fuzzy(
                str(e.get("attribute", "")), case["attribute"]
            ):
                if e.get("asserts"):
                    asserted.append(str(e["asserts"]))
                if e.get("retracts"):
                    killed.add(str(e["retracts"]))
    reader_answer = resolve_cell(asserted, killed)

    mem_score = score_answer(case, memory_answer)
    reader_score = score_answer(case, reader_answer)
    return {
        "id": case["id"],
        "family": case["family"],
        "memory_answer": memory_answer,
        "reader_answer": reader_answer,
        "memory": mem_score,
        "reader": reader_score,
        "parsed_edges": len(edges),
    }


def summarize(rows: list[dict]) -> dict:
    fam = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        f = r["family"]
        fam[f]["n"] += 1
        fam[f]["memory_correct"] += int(r["memory"]["correct"])
        fam[f]["reader_correct"] += int(r["reader"]["correct"])
        fam[f]["memory_leaked"] += int(r["memory"]["leaked"])
        fam[f]["reader_leaked"] += int(r["reader"]["leaked"])
        fam[f]["memory_abstained"] += int(r["memory"]["abstained"])
        fam[f]["reader_abstained"] += int(r["reader"]["abstained"])

    by_family = {}
    for f, c in sorted(fam.items()):
        n = c["n"]
        by_family[f] = {
            "n": n,
            "memory_accuracy": round(c["memory_correct"] / n, 4),
            "reader_accuracy": round(c["reader_correct"] / n, 4),
            "memory_leak": round(c["memory_leaked"] / n, 4),
            "reader_leak": round(c["reader_leaked"] / n, 4),
            "memory_abstain": round(c["memory_abstained"] / n, 4),
            "reader_abstain": round(c["reader_abstained"] / n, 4),
        }

    a_rows = [r for r in rows if r["family"] == "A"]
    b = sum(1 for r in a_rows if r["memory"]["correct"] and not r["reader"]["correct"])
    c = sum(1 for r in a_rows if not r["memory"]["correct"] and r["reader"]["correct"])
    return {"by_family": by_family, "family_a_memory_only_right": b, "family_a_reader_only_right": c}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/cases.jsonl")
    p.add_argument("--families", default="A,C")
    p.add_argument("--chunk", type=int, default=12)
    p.add_argument("--out", default="data/confirm_deterministic_full.json")
    args = p.parse_args()

    families = {f.strip() for f in args.families.split(",") if f.strip()}
    cases = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [run_case(case, args.chunk) for case in cases if case["family"] in families]
    payload = {
        "kind": "deterministic_visible_text_confirmation",
        "data": args.data,
        "families": sorted(families),
        "chunk": args.chunk,
        "summary": summarize(rows),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
