# CORE-RT: Corrective Resolution under Equal-Compute Read-Time Map-Reduce

CORE-RT is a benchmark for one narrow question:

> If notes contain explicit corrections, does resolving those corrections once at write time and caching
> the result answer current-value questions more accurately than an equal-compute read-time correction
> resolver operating on the same notes?

The control is the point. The read-time baseline is not weak RAG. It sees the same prose, uses the
same correction rules, and follows the same map-reduce compute pattern. The only variable is where the
resolution happens: write time versus read time.

Current artifact status: under these controlled conditions, there is no evidence that persisted
write-time belief revision improves current-value accuracy over a read-time resolver operating on the
same evidence. With perfect edges the deterministic resolver is flawless; with weak extraction memory
pays an extraction tax; with a strong extractor memory matches the read-time reader on the tested
confirming cases.

## What Is In This Repo

- `SPEC.md`: benchmark contract, conflict families, systems, metrics, and kill criteria.
- `generate.py`: synthetic novel-string case generator with stable deterministic seeding.
- `audit.py`: anti-tautology canary checks for recency, frequency, and last-line shortcuts.
- `run.py`: plain RAG, single-window, read-time map-reduce, memory, and closed-book harnesses.
- `run_live_confirm.py`: live model controller for paired memory vs map-reduce confirmation runs.
- `run_batched_live_confirm.py`: batched live confirmation runner that reduces model call count.
- `score.py`: deterministic scorer and tautology canaries.
- `score_confirm.py`: scorer for the external-LLM confirming run.
- `confirm_deterministic.py`: full-size deterministic visible-text confirming sweep.
- `analyze.py`: family metrics and paired McNemar analysis.
- `slope_retrieval.py`: LLM-free long-context retrieval-axis slope sweep.
- `LIMITATIONS.md`: scope boundaries and claims this benchmark does not make.
- `data/`: generated cases, canary pilot, confirming artifacts, and slope artifact.

## Key Results

CORE-RT correction family:

- Oracle edges: memory resolves at `1.00`.
- Weak extractor: memory collapses on family A, showing the extraction tax.
- Strong extractor: memory and read-time reader both score `1.00` on the confirming sample, with McNemar `p = 1.000`.
- Full deterministic visible-text confirmation over all A/C cases: memory and read-time reader both score `1.00` on family A (`n=60`) and family C (`n=60`), with zero family-A poison leak and no paired advantage either way.

Long-context retrieval-axis sweep:

- Command:
  `python slope_retrieval.py --per-family 30 --Ns 30,120,480,1920,5000 --ks 4,10,20,40 --seed 7 --out data/slope_retrieval.json`
- At `k=10`, `k=20`, and `k=40`, all tested policies remain at `1.00` accuracy and `0.00` poison leak through `N=5000`.
- At `k=4`, failures are retrieval-budget starvation rather than an N-slope effect.
- Full rows are in `data/slope_retrieval.json`; this sweep is a regression/control for retrieval slope, not the headline result.
- This is not a claim that retrieval never degrades with scale. It is a claim about this benchmark's cell-scoped setup and tested reader budgets.

## Reproduce

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the canary audit:

```bash
python audit.py data/pilot.jsonl
```

Score the confirming run:

```bash
python score_confirm.py data/confirm_results.json
```

Run the full deterministic visible-text confirmation:

```bash
python confirm_deterministic.py --data data/cases.jsonl --families A,C --out data/confirm_deterministic_full.json
```

Run a live paired A/C confirmation when a model key is available:

```bash
python run_live_confirm.py --data data/cases.jsonl --families A,C --limit-per-family 20 --out-prefix data/confirm_ac_live --workers 1 --chunk 999
```

`run_live_confirm.py` reads `GEMINI_API_KEY` or `OPENAI_API_KEY` from the process environment. Do not write keys into repo files.

For quota-constrained live runs, use the batched runner:

```bash
python run_batched_live_confirm.py --data data/cases.jsonl --families A,C --limit-per-family 60 --reader-batch-size 4 --memory-batch-size 2 --sleep 1.5 --out-prefix data/confirm_ac_batched_live
```

Run the retrieval-axis slope sweep:

```bash
python slope_retrieval.py --per-family 30 --Ns 30,120,480,1920,5000 --ks 4,10,20,40 --seed 7 --out data/slope_retrieval.json
```

## Interpretation

CORE-RT isolates the core claim from timestamp shortcuts, recency shortcuts, and parametric recall. The benchmark provides no evidence of an accuracy advantage for write-time structured memory over a fair read-time reader.

That does not make persisted memory pointless. In the tested explicit corrective setting, the surviving case is narrower: auditability, structural provenance, defensible abstention, and economics in high-query settings. Memory pays resolution cost once, then reads a cached cell; a read-time resolver repeats reduction for each query.

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

CORE-RT therefore falsifies an accuracy advantage under the tested controls, while leaving an engineering cost advantage possible when the same evidence is queried repeatedly.
