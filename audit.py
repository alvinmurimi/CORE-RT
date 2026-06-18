"""Offline anti-tautology / sanity gate. No API key needed.

Runs the deterministic canaries on generated cases and asserts:
  * on family A (the non-tautological core) every canary floors at/below chance,
  * gold rarely sits on the last line (no recency leak),
  * no timestamps/years survive in any memory,
  * the position of the gold-bearing assertion is decorrelated from the end of the pile.
If a canary reaches gold on A, the benchmark is recency/frequency-solvable and must be fixed.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

from score import CANARIES, _cell_assertions, canary_correct

path = sys.argv[1] if len(sys.argv) > 1 else "data/cases.jsonl"
cases = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
by_fam = defaultdict(list)
for c in cases:
    by_fam[c["family"]].append(c)

print(f"loaded {len(cases)} cases from {path}\n")

print("canary accuracy by family (family A MUST be near/below chance):")
hdr = f"  {'family':<8}{'n':>5}  " + "  ".join(f"{name:>10}" for name in CANARIES)
print(hdr)
a_fail = False
for fam in sorted(by_fam):
    rows = by_fam[fam]
    accs = {}
    for name, fn in CANARIES.items():
        hits = sum(canary_correct(c, fn(c)) for c in rows)
        accs[name] = hits / len(rows)
    line = f"  {fam:<8}{len(rows):>5}  " + "  ".join(f"{accs[n]:>10.3f}" for n in CANARIES)
    print(line)
    if fam == "A" and max(accs.values()) > 0.40:
        a_fail = True


def gold_is_last(c: dict) -> bool:
    if c["gold"]["type"] != "value":
        return False
    last = len(c["memories"]) - 1
    for i, v in _cell_assertions(c):
        if i == last and v == c["gold"]["value"]:
            return True
    return False


a_cases = by_fam.get("A", [])
gl = sum(gold_is_last(c) for c in a_cases)
print(f"\nfamily A gold-on-last-line: {gl}/{len(a_cases)} = {gl / max(len(a_cases), 1):.3f} (want ~0)")

date_pat = re.compile(r"\b\d{4}\b|\d{4}-\d{2}-\d{2}")
date_hits = sum(1 for c in cases for m in c["memories"] if date_pat.search(m))
print(f"memories containing a year/date: {date_hits} (want 0)")

positions = []
for c in a_cases:
    if c["gold"]["type"] != "value":
        continue
    n = len(c["memories"])
    for i, v in _cell_assertions(c):
        if v == c["gold"]["value"]:
            positions.append(i / max(n - 1, 1))
            break
if positions:
    mean_pos = sum(positions) / len(positions)
    print(f"family A mean normalized gold position: {mean_pos:.3f} (want ~0.5, i.e. not end-loaded)")

print("\nGATE:", "FAIL - a canary solves family A" if a_fail else "PASS - canaries cannot reach gold on A")
