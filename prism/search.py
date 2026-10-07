"""Search orchestration (Member C)."""
from __future__ import annotations

import heapq

import numpy as np

from prism.boolean_ops import phrase_match
from prism.authority import recency_scores
from prism.candidates import candidates
from prism.config import INDEX_DIR, VARIANTS
from prism.authority import recency_scores
from prism.gate import adaptive_weights, beta_of_q, specificity
from prism.index import Index
from prism.parser import parse
from prism.schema import Result
from prism.scoring import bm25_scores, cosine_scores

_INDEX: Index | None = None
_AUTHORITY_CACHE: dict[int, dict[str, np.ndarray]] = {}
_RECENCY_CACHE: dict[int, np.ndarray] = {}
_CITATION_PRESENT_CACHE: dict[int, bool] = {}


def set_index(index: Index) -> None:
    """Set the active index (useful for embedding callers and small corpora)."""
    global _INDEX
    _INDEX = index
    _AUTHORITY_CACHE.clear()
    _RECENCY_CACHE.clear()
    _CITATION_PRESENT_CACHE.clear()


def _authority_array(index: Index, mode: str) -> np.ndarray:
    """Cache the index's scalar authority accessor as an aligned NumPy array."""
    cache = _AUTHORITY_CACHE.setdefault(id(index), {})
    if mode not in cache:
        cache[mode] = np.fromiter(
            (index.authority(doc, mode) for doc in range(index.n_docs)),
            dtype=np.float64,
            count=index.n_docs,
        )
    return cache[mode]


def _recency_array(index: Index) -> np.ndarray:
    key = id(index)
    if key not in _RECENCY_CACHE:
        years = [index.field_value(doc, "year") for doc in range(index.n_docs)]
        _RECENCY_CACHE[key] = recency_scores(years)
    return _RECENCY_CACHE[key]


def _has_citation_counts(index: Index) -> bool:
    key = id(index)
    if key not in _CITATION_PRESENT_CACHE:
        _CITATION_PRESENT_CACHE[key] = bool(np.any(_authority_array(index, "raw")))
    return _CITATION_PRESENT_CACHE[key]


def _get_index() -> Index:
    global _INDEX
    if _INDEX is None:
        if not INDEX_DIR.exists():
            raise FileNotFoundError(f"Index not found at {INDEX_DIR}; build it with scripts/build_index.py")
        _INDEX = Index.load(INDEX_DIR)
    return _INDEX


def _components(pq, index: Index, cfg, use_champions: bool | None = None):
    champion_mode = cfg.champions if use_champions is None else use_champions
    cand = candidates(pq, index, use_champions=champion_mode)
    if cfg.scorer == "bm25":
        scores = bm25_scores(index, pq.terms, cand)
        return {}, scores, np.zeros(index.n_docs), 0.0, {}, cand

    zone_scores = {zone: cosine_scores(index, zone, pq.terms, cand) for zone in cfg.zone_weights}
    weights = adaptive_weights(index, pq.terms, cfg.zone_weights) if cfg.adaptive_zones else dict(cfg.zone_weights)
    beta = beta_of_q(specificity(index, pq.terms), cfg.beta, cfg.s_lo, cfg.s_hi) if cfg.gate else cfg.beta
    mode = cfg.authority_mode
    if mode == "none":
        authority = np.zeros(index.n_docs, dtype=np.float64)
    else:
        authority = _authority_array(index, mode)
        # V5-V7's cohort array has a nonzero minimum rank even when every
        # citation count is missing. The raw array distinguishes that case.
        if not _has_citation_counts(index):
            authority = _recency_array(index)
    total = np.zeros(index.n_docs, dtype=np.float64)
    for zone, values in zone_scores.items():
        total += float(weights[zone]) * values
    # Authority may reorder a text match, but must never introduce a non-match.
    text_match = np.zeros(index.n_docs, dtype=bool)
    for values in zone_scores.values():
        text_match |= values > 0
    total += beta * authority * text_match
    if cfg.phrase_boost > 0 and pq.phrases:
        for phrase in pq.phrases:
            matched = phrase_match(index, phrase, "all")
            total[matched] += cfg.phrase_boost
    return zone_scores, total, authority, beta, weights, cand


def score_components(query: str):
    """Return (per-zone cosine arrays, selected authority array) for tuning."""
    index = _get_index()
    pq = parse(query)
    cfg = VARIANTS["V5"]
    zone_scores, _, authority, _, _, _ = _components(pq, index, cfg, use_champions=False)
    return zone_scores, authority


def search(query: str, k: int = 10, variant: str = "V2") -> list[Result]:
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    if k < 0:
        raise ValueError("k must be nonnegative")
    if k == 0:
        return []
    index = _get_index()
    cfg = VARIANTS[variant]
    pq = parse(query)
    zone_scores, scores, authority, beta, weights, cand = _components(pq, index, cfg)

    # If champion truncation leaves too few matches, retry against full postings.
    text_mask = np.zeros(index.n_docs, dtype=bool)
    if cfg.scorer == "bm25":
        text_mask = scores > 0
    else:
        for values in zone_scores.values():
            text_mask |= values > 0
    if cfg.champions and int(text_mask.sum()) < k:
        zone_scores, scores, authority, beta, weights, cand = _components(pq, index, cfg, use_champions=False)
        text_mask = scores > 0 if cfg.scorer == "bm25" else np.logical_or.reduce([v > 0 for v in zone_scores.values()])

    eligible = np.flatnonzero(text_mask)
    ranked = heapq.nlargest(min(k, len(eligible)), eligible.tolist(), key=lambda doc: (float(scores[doc]), -doc))
    results = []
    for doc in ranked:
        zscores = {zone: float(values[doc]) for zone, values in zone_scores.items()}
        results.append(Result(
            doc_id=index.doc_id(doc), score=float(scores[doc]), zone_scores=zscores,
            authority=float(authority[doc]), beta=float(beta), matched_terms=list(dict.fromkeys(pq.terms)),
            title=index.title(doc),
        ))
    return results
