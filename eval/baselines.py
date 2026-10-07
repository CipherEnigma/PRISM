"""D: independent placeholder baselines used only to validate the harness, never as the engine.

These use scikit-learn / rank_bm25 on a corpus slice, so they cross-check the pipeline
(loader -> runner -> metrics) before A's index exists and later sanity-check our own scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass
class Hit:
    doc_id: str
    score: float


def tfidf_search_fn(docs: Mapping[str, str]):
    """search(query, k, variant) over {doc_id: text} using scikit-learn TfidfVectorizer + cosine."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    ids = sorted(docs)
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform([docs[i] for i in ids])

    def search(query: str, k: int = 10, variant: str = "tfidf") -> list[Hit]:
        scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
        order = np.lexsort((np.arange(len(ids)), -scores))[:k]     # ties: smaller index first
        return [Hit(ids[i], float(scores[i])) for i in order if scores[i] > 0]

    return search


def bm25_search_fn(docs: Mapping[str, str]):
    """search(query, k, variant) over {doc_id: text} using rank_bm25 (test cross-check only)."""
    from rank_bm25 import BM25Okapi

    ids = sorted(docs)
    bm25 = BM25Okapi([docs[i].lower().split() for i in ids])

    def search(query: str, k: int = 10, variant: str = "bm25") -> list[Hit]:
        scores = bm25.get_scores(query.lower().split())
        order = np.lexsort((np.arange(len(ids)), -scores))[:k]
        return [Hit(ids[i], float(scores[i])) for i in order if scores[i] > 0]

    return search


def top_overlap(a: Sequence[str], b: Sequence[str], k: int = 10) -> float:
    """Share of the top-k of `a` that also appears in the top-k of `b` (a bug check, not a metric).

    Divides by the number of documents `a` actually returned (at most k), so short result
    lists are not penalized. Two empty lists agree (1.0); one empty and one not do not (0.0).
    """
    top_a, top_b = list(a)[:k], set(list(b)[:k])
    if not top_a:
        return 1.0 if not top_b else 0.0
    return sum(1 for d in top_a if d in top_b) / len(top_a)
