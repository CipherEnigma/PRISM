"""Query specificity, authority gating (N2), and adaptive zone weights (N3)."""
from __future__ import annotations

import math


def specificity(index, terms: list[str]) -> float:
    """Mean normalized idf of query terms in [0, 1]."""
    n_docs = index.n_docs
    if n_docs <= 1:
        return 0.0
    denominator = math.log10(n_docs)
    values = []
    for term in terms:
        df = index.df(term, "all")
        if df > 0:
            values.append(min(1.0, max(0.0, math.log10(n_docs / df) / denominator)))
    return sum(values) / len(values) if values else 0.0


def beta_of_q(s: float, beta_max: float, s_lo: float = 0.2, s_hi: float = 0.8) -> float:
    """Linearly lower authority weight as query specificity increases."""
    if beta_max < 0:
        raise ValueError("beta_max must be nonnegative")
    if s_hi <= s_lo:
        raise ValueError("s_hi must be greater than s_lo")
    fraction = min(1.0, max(0.0, (s_hi - s) / (s_hi - s_lo)))
    return beta_max * fraction


def adaptive_weights(index, terms: list[str], global_weights: dict[str, float], alpha: float = 0.5) -> dict[str, float]:
    """Blend global zone weights with the query's normalized zone idf mass."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    zones = list(global_weights)
    total_global = sum(max(0.0, float(global_weights[z])) for z in zones)
    if total_global <= 0:
        raise ValueError("global zone weights must have a positive sum")
    global_norm = {z: max(0.0, float(global_weights[z])) / total_global for z in zones}
    n_docs = index.n_docs
    denom = math.log10(n_docs) if n_docs > 1 else 1.0
    counts = {z: 0.0 for z in zones}
    for zone in zones:
        for term in set(terms):
            df = index.df(term, zone)
            if df > 0:
                counts[zone] += math.log10(n_docs / df) / denom if n_docs > 1 else 0.0
    total = sum(counts.values())
    if total <= 0:
        return global_norm
    return {z: (1.0 - alpha) * global_norm[z] + alpha * counts[z] / total for z in zones}
