"""Score the external-LLM confirming run. Input: the workflow result JSON (a list of {id, extract, reader}).
Routes the extracted edges through the SAME deterministic resolver memory uses, scores memory and the
reader against provenance-gold, and runs the paired McNemar on family A."""
import collections
import json
import sys

from scipy.stats import binomtest

from run import _fuzzy, resolve_cell
from score import score_answer

cases = {json.loads(l)["id"]: json.loads(l)
         for l in open("data/pilot.jsonl", encoding="utf-8") if l.strip()}
results = json.load(open(sys.argv[1], encoding="utf-8"))

fam = collections.Counter()
mem_ok = collections.Counter()
rd_ok = collections.Counter()
mem_abs = collections.Counter()
rd_abs = collections.Counter()
rows = []
for r in results:
    if not r or r.get("id") not in cases:
        continue
    c = cases[r["id"]]
    edges = (r.get("extract") or {}).get("edges") or []
    asserted, killed = [], set()
    for e in edges:
        if not isinstance(e, dict):
            continue
        if _fuzzy(str(e.get("subject", "")), c["entity"]) and _fuzzy(str(e.get("attribute", "")), c["attribute"]):
            if e.get("asserts"):
                asserted.append(str(e["asserts"]))
            if e.get("retracts"):
                killed.add(str(e["retracts"]))
    mem_ans = resolve_cell(asserted, killed)
    ms, rs = score_answer(c, mem_ans), score_answer(c, r.get("reader", ""))
    fam[c["family"]] += 1
    mem_ok[c["family"]] += ms["correct"]
    rd_ok[c["family"]] += rs["correct"]
    mem_abs[c["family"]] += ms["abstained"]
    rd_abs[c["family"]] += rs["abstained"]
    rows.append((r["id"], c["family"], mem_ans, ms["correct"], r.get("reader", ""), rs["correct"]))

print("CONFIRMING RUN (external LLM) - structured memory (extract+resolve) vs read-time reader")
print(f"{'family':<8}{'n':>4}{'mem_acc':>9}{'read_acc':>9}{'mem_abst':>9}{'read_abst':>10}")
for f in sorted(fam):
    n = fam[f]
    print(f"{f:<8}{n:>4}{mem_ok[f] / n:>9.2f}{rd_ok[f] / n:>9.2f}{mem_abs[f] / n:>9.2f}{rd_abs[f] / n:>10.2f}")

A = [r for r in rows if r[1] == "A"]
b = sum(1 for r in A if r[3] and not r[5])
c2 = sum(1 for r in A if not r[3] and r[5])
p = binomtest(min(b, c2), b + c2, 0.5).pvalue if (b + c2) else 1.0
print(f"\nfamily A paired McNemar: memory-only-right={b}  reader-only-right={c2}  p={p:.3f}")
print("\nper-case (id, family, mem_answer, mem_ok, reader_answer, read_ok):")
for r in rows:
    print("  ", r)
