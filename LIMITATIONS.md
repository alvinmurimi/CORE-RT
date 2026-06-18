# Limitations

CORE-RT is intentionally narrow. It is a benchmark for explicit corrective current-value resolution under equal evidence and equal compute, not a general benchmark for all memory, retrieval, or agent behavior.

## What CORE-RT Does Not Claim

- It does not claim persisted memory is useless.
- It does not claim retrieval never degrades with scale.
- It does not claim the result holds for all models, prompts, corpora, or retrievers.
- It does not claim synthetic explicit corrections cover naturally occurring memory streams.
- It does not measure latency, storage cost, update cost, or amortized query economics directly.

## Synthetic Data

The data is synthetic by design. Novel entities and values remove parametric recall, hidden provenance gives unambiguous gold labels, and missing timestamps force systems to use correction semantics instead of date sorting.

That isolation improves internal validity, but weakens external validity. Real-world memory streams include aliases, partial corrections, uncertain sources, document-level context, implicit drift, and natural timestamps.

## Live Model Coverage

The live external-LLM confirmation is small. It supports the current artifact's equality pattern, but it is not a broad cross-model study. A stronger conference submission should add larger live runs across multiple model families.

The deterministic visible-text confirmation is larger, but it is grammar-aware and not an LLM result. It checks benchmark logic and the equal-compute comparison under faithful extraction.

## Retrieval-Axis Sweep

The retrieval-axis sweep is a control, not the headline result. It shows that the tested cell-scoped and hybrid evidence policies remain flat through 5,000 notes at the tested reader budgets.

It should not be read as proof that retrieval is scale-invariant. It says this benchmark's cell-scoped retrieval setup does not create a hidden slope advantage for memory under the tested conditions.

## Practical Memory Value

CORE-RT falsifies an intrinsic accuracy-superiority claim only in the tested explicit corrective setting. Persisted memory may still be valuable for provenance, auditability, defensible abstention, latency, and amortized cost when the same evidence is queried repeatedly.
