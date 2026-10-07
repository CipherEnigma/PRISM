"""D: paired significance tools for ~25 topics. Per-topic metric lists are aligned by topic."""
from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

import numpy as np


def paired_diffs(per_topic_a: Mapping[str, float], per_topic_b: Mapping[str, float],
                 qids: Sequence[str] | None = None) -> np.ndarray:
    """a - b for each topic present in both, in sorted topic order (or the given order)."""
    qids = sorted(set(per_topic_a) & set(per_topic_b)) if qids is None else list(qids)
    return np.array([per_topic_a[q] - per_topic_b[q] for q in qids], dtype=np.float64)


def bootstrap_ci(diffs: Sequence[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Mean difference with a percentile bootstrap CI over topics: (mean, low, high)."""
    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(low), float(high)


def sign_test(diffs: Sequence[float]) -> dict:
    """Exact two-sided sign test. Ties (zero differences) are dropped, as is standard."""
    d = np.asarray(diffs, dtype=np.float64)
    wins, losses, ties = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    n = wins + losses
    if n == 0:
        return {"wins": wins, "losses": losses, "ties": ties, "p": 1.0}
    tail = sum(math.comb(n, i) for i in range(0, min(wins, losses) + 1)) / 2 ** n
    return {"wins": wins, "losses": losses, "ties": ties, "p": min(1.0, 2 * tail)}


def holm_correct(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control), keyed like the input."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def _ranks(values: Sequence[float]) -> np.ndarray:
    """Average ranks (ties share the mean rank), starting at 1."""
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=np.float64)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (Pearson on average ranks); 0 if either side is constant."""
    rx, ry = _ranks(x), _ranks(y)
    if len(rx) < 2 or rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_ci(x: Sequence[float], y: Sequence[float], n_boot: int = 5000, alpha: float = 0.05,
                seed: int = 0) -> tuple[float, float, float]:
    """Spearman rho with a bootstrap CI over (x, y) pairs: (rho, low, high)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rhos.append(spearman(x[idx], y[idx]))
    low, high = np.percentile(rhos, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return spearman(x, y), float(low), float(high)


def compare_to_baseline(per_topic: Mapping[str, Mapping[str, float]], baseline: str, metric: str,
                        qids: Sequence[str], n_boot: int = 10000) -> dict[str, dict]:
    """For every variant vs the baseline on `metric`: mean diff, CI, wins/losses, raw and Holm p."""
    rows: dict[str, dict] = {}
    for name, topics in per_topic.items():
        if name == baseline:
            continue
        valid = [q for q in qids if q in topics and q in per_topic[baseline]]
        diffs = paired_diffs({q: topics[q][metric] for q in valid},
                             {q: per_topic[baseline][q][metric] for q in valid}, valid)
        mean, low, high = bootstrap_ci(diffs, n_boot=n_boot)
        rows[name] = {"mean_diff": mean, "ci_low": low, "ci_high": high, **sign_test(diffs)}
    adjusted = holm_correct({k: v["p"] for k, v in rows.items()})
    for name, p_holm in adjusted.items():
        rows[name]["p_holm"] = p_holm
    return rows


def repeated_kfold(qids: Sequence[str], k: int = 5, repeats: int = 10,
                   seed: int = 0) -> list[tuple[list[str], list[str]]]:
    """(train, test) topic splits: `repeats` random k-fold partitions. A robustness check
    next to the fixed 25/25 split, never a replacement for it."""
    ordered = sorted(qids)
    folds = []
    for r in range(repeats):
        shuffled = ordered[:]
        random.Random(seed + r).shuffle(shuffled)
        for i in range(k):
            test = sorted(shuffled[i::k])
            train = sorted(q for q in ordered if q not in set(test))
            folds.append((train, test))
    return folds
