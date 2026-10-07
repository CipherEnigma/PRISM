"""Term-at-a-time lnc.ltc cosine and BM25 scoring (Member C)."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np


def cosine_scores(index, zone: str, terms: list[str], candidates=None) -> np.ndarray:
    """Return lnc.ltc cosine scores aligned with internal document ids."""
    n_docs = index.n_docs
    scores = np.zeros(n_docs, dtype=np.float64)
    if n_docs <= 0:
        return scores

    q_weights: dict[str, float] = {}
    for term, tf in Counter(terms).items():
        df = index.df(term, zone)
        if df > 0:
            q_weights[term] = (1.0 + math.log10(tf)) * math.log10(n_docs / df)
    q_norm = math.sqrt(sum(weight * weight for weight in q_weights.values())) or 1.0

    for term, weight in q_weights.items():
        docs, tfs = index.postings(term, zone)
        if len(docs):
            scores[docs] += (weight / q_norm) * (1.0 + np.log10(tfs))

    norms = index.norms(zone)
    nonzero = norms > 0
    scores[nonzero] /= norms[nonzero]
    scores[~nonzero] = 0.0
    if candidates is not None:
        allowed = np.zeros(n_docs, dtype=bool)
        allowed[np.asarray(candidates, dtype=np.int64)] = True
        scores[~allowed] = 0.0
    return scores


def bm25_scores(index, terms: list[str], candidates=None, k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """Return Robertson BM25 scores over the all zone."""
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("k1 must be positive and b must be between 0 and 1")
    n_docs = index.n_docs
    scores = np.zeros(n_docs, dtype=np.float64)
    if n_docs <= 0:
        return scores
    lengths = index.lengths("all")
    avg_len = index.avg_len("all")
    if avg_len <= 0:
        return scores

    for term, qtf in Counter(terms).items():
        df = index.df(term, "all")
        if df <= 0:
            continue
        idf = math.log1p((n_docs - df + 0.5) / (df + 0.5))
        docs, tfs = index.postings(term, "all")
        doc_lengths = lengths[docs]
        denom = tfs + k1 * (1.0 - b + b * doc_lengths / avg_len)
        scores[docs] += idf * tfs * (k1 + 1.0) / denom
    if candidates is not None:
        allowed = np.zeros(n_docs, dtype=bool)
        allowed[np.asarray(candidates, dtype=np.int64)] = True
        scores[~allowed] = 0.0
    return scores
