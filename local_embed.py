"""Offline dense embeddings via LSA (TF-IDF + truncated SVD). No API, no rate limit, deterministic.

Fit once on the full corpus, then transform any text into an L2-normalized dense vector that drops
straight into the cosine-similarity adapters (dot product of two unit vectors is their cosine). The
point is a controlled, reproducible retriever: identical vectors every run, so a RAG-vs-memory
comparison is never confounded by embedding drift or a rate-limited API.

The vectors are classic LSA, not a neural embedder. That is fine here: every system under test shares
this exact retriever, so the absolute retrieval strength cancels out and only the delta between systems
-- the thing we are measuring -- remains.
"""

from __future__ import annotations

import random

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class LocalEmbedder:
    """Corpus-fit LSA embedder. `embed(texts)` returns unit-norm dense vectors as plain lists, cached
    by text so a repeated statement (the shared distractor pool) is transformed once."""

    def __init__(self, corpus: list[str], dims: int = 256, seed: int = 0, fit_cap: int = 25_000):
        docs = [c for c in dict.fromkeys(corpus) if c and c.strip()]
        if not docs:
            raise ValueError("LocalEmbedder needs a non-empty corpus to fit on.")
        # Fitting the SVD on the whole corpus is needless and pathologically slow on large suites; a
        # seeded sample gives an equivalent basis, and transform still runs on every text afterwards.
        if len(docs) > fit_cap:
            docs = random.Random(seed).sample(docs, fit_cap)
        self.vec = TfidfVectorizer(lowercase=True, stop_words="english", min_df=1,
                                   ngram_range=(1, 2), max_features=200_000)
        X = self.vec.fit_transform(docs)
        # SVD components must be < min(n_docs, n_features); clamp so tiny corpora still fit.
        n_comp = max(2, min(dims, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=seed)
        self.svd.fit(X)
        self.dims = n_comp
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        misses = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if misses:
            Z = normalize(self.svd.transform(self.vec.transform(misses)))
            for t, row in zip(misses, Z, strict=False):
                self._cache[t] = [float(x) for x in row]
        return [self._cache[t] for t in texts]
