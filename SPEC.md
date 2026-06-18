# CORE-RT: Corrective Resolution under Equal-Compute Read-Time Map-Reduce

Timestamp-free, provenance-gold benchmark for the question: **can persisted write-time belief revision
answer current-value queries more accurately than an equal-compute read-time correction resolver,
without timestamps?**

Locked by an adversarial design pass. This file is the contract the code implements.

## Thesis

Persisted write-time belief revision shows an accuracy advantage only if it out-resolves a read-time
map-reduce reader that has the same prose, same LLM, and same compute pattern. The decisive variable
is resolve-once-and-cache at write time versus resolve-by-re-derivation at read time under the same
evidence and resolution budget.

## Data

- Novel fictional entities and values make closed-book parametric recall impossible by construction.
- No timestamps and no validity windows. Only ingestion order exists, and it is decorrelated from gold.
- Entity is named in every sentence. Unnamed corrections are a retrieval artifact, not reasoning.
- One `(entity, attribute)` cell per case.

### Conflict Families

- **A - explicit correction**: `V_true` is asserted early and once; `V_poison` is asserted more often
  with `k` restatements; one natural-language retraction negates `V_poison`. Gold is `V_true`, the
  oldest and least repeated value. Half are negation-only.
- **B - no-correction control**: A's surface without the retraction. Gold is the dominant value. This
  catches systems that win A/C by reflexively distrusting recency.
- **C - unresolvable**: two mutually exclusive values with no tiebreaker, or two contradictory
  retractions. Gold is abstain.
- **D - re-update**: retract `V_poison`, then validly reassert `V_new`. Gold is `V_new`. This family is
  excluded from the superiority claim because recency can solve it.

### Burial

Distractors include same-entity/different-attribute lines, verbatim restatements of the retracted value
on other attributes, and same-attribute corrections about other entities. None assert the queried
`(entity, attribute)` cell.

### Gold by Hidden Provenance

Every assertion carries a hidden role (`original`, `poison`, `retraction`, `reupdate`, or `distractor`)
used only by the scorer, never by the system. Gold is computed from provenance, so it is not the most
recent, most frequent, or last-mentioned value.

## Systems

1. **Plain RAG**: top-k over raw sentences, no correction instruction, no abstain. This is a floor.
2. **Single-pass metadata RAG, k=ALL**: whole case in one window, told to honor corrections and abstain.
   This is a calibration strawman, not the decisive baseline.
3. **Read-time map-reduce reader**: same LLM, temperature 0, folds chunks into a running cell with the
   same resolution rules memory uses at write time. Memory must beat this.
4. **Provenance-faithful belief-revision memory**: derives retraction edges from prose using the same
   LLM, routes them per cell, folds assertions into one cached cell at write time, and reads at k=1.
5. **Canaries**: deterministic non-LLM audit policies: most-recent-by-order, most-frequent, and
   last-line. They must floor at or below chance on family A.
6. **Closed-book probe**: empty store. It must score near zero on novel strings.

## Metrics

- Per-family accuracy.
- Poison leak.
- Abstention correctness on family C.
- Accuracy and leak slope versus reader budget `k` and distractor count `N`.
- Write-time edge-derivation precision and recall on held-out cues.
- Paired McNemar test for memory versus map-reduce.

## Non-Accuracy Tradeoff

A negative accuracy result does not imply persisted memory is useless. If the same corpus is queried
many times, memory can trade write-time work for cheaper repeated reads.

Let \(Q\) be the number of repeated queries over the same evidence:

$$
C_{\mathrm{memory}} = C_{\mathrm{write}} + Q C_{\mathrm{cached\_read}}
$$

$$
C_{\mathrm{read}} = Q C_{\mathrm{read\_reduce}}
$$

Memory amortizes when:

$$
Q > \frac{C_{\mathrm{write}}}{C_{\mathrm{read\_reduce}} - C_{\mathrm{cached\_read}}}
$$

CORE-RT treats that as an engineering and economics claim, not as evidence of an accuracy advantage.

## Kill Criteria

Abandon the accuracy thesis if any hold at adequate `n`:

1. Map-reduce matches memory on family-A accuracy.
2. Memory and map-reduce family-A slopes versus `k` or distractor count are statistically
   indistinguishable.
3. Under symmetric context, memory poison leak is not significantly below map-reduce poison leak.
4. Write-time edge-derivation error is approximately map-reduce per-query error, making memory
   map-reduce relabeled.
5. A canary wins where memory wins, or closed-book scores non-trivially on family A.

If only family C survives, the honest finding is narrow: no current-value accuracy edge over an
equal-compute reader on explicit corrections; the only edge is calibrated abstention, and even that is
voided if a map-reduce reader given the same hard-abstain scaffold matches it.
