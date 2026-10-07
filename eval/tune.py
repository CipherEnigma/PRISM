"""D: grid search on the tuning half only, reusing cached score components.

The net score is linear in the per-zone cosine arrays and the authority array,
    net = sum_z w_z * cos_z + beta * g * [document matches the text query]
(the same form as prism/search.py), so each topic's arrays are computed once and every grid
point only re-weights them in memory. No grid point touches the index.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from eval.baselines import top_overlap
from eval.metrics import evaluate
from eval.stats import repeated_kfold

RankFn = Callable[[str, dict], list[str]]      # (qid, params) -> ranked doc ids, best first


@dataclass
class TopicComponents:
    """Everything one topic needs to be re-scored: per-zone cosines (candidate-restricted
    already) and the query's specificity (only needed by the gate)."""
    zone_scores: dict[str, np.ndarray]
    specificity: float = 0.0


@dataclass
class GridRow:
    params: dict
    mean: dict[str, float]                       # mean of every metric over the tuned topics
    per_topic: dict[str, dict[str, float]] = field(default_factory=dict)


def collect_components(queries: Mapping[str, str], qids: Sequence[str],
                       zone_fn: Callable[[str], dict[str, np.ndarray]],
                       specificity_fn: Callable[[str], float] | None = None) -> dict[str, TopicComponents]:
    """Compute each topic's components once. `zone_fn(query)` returns {zone: cosine array}."""
    return {q: TopicComponents(zone_fn(queries[q]), specificity_fn(queries[q]) if specificity_fn else 0.0)
            for q in qids}


def assert_tuning_only(qids: Sequence[str], split: Mapping) -> None:
    """Refuse to tune on any topic outside the tuning half (keeps the reporting half clean)."""
    leaked = sorted(set(qids) - set(split["tune"]))
    if leaked:
        raise ValueError(f"tuning would touch reporting topics {leaked}; tune on split['tune'] only")


def top_k_ids(scores: np.ndarray, eligible: np.ndarray, k: int) -> np.ndarray:
    """Internal ids of the k best eligible documents: score descending, smaller id first on ties
    (the same order as search())."""
    ids = np.flatnonzero(eligible)
    if ids.size == 0:
        return ids
    vals = scores[ids]
    if ids.size > k:
        threshold = np.partition(vals, -k)[-k]
        keep = vals >= threshold                  # keeps every document tied at the cut
        ids, vals = ids[keep], vals[keep]
    return ids[np.lexsort((ids, -vals))][:k]


def linear_ranked(comp: TopicComponents, weights: Mapping[str, float], beta: float,
                  authority: np.ndarray, doc_ids: Sequence[str], k: int = 100) -> list[str]:
    """Rank one topic from cached components, mirroring prism.search's net score."""
    total = np.zeros_like(authority, dtype=np.float64)
    eligible = np.zeros(authority.shape, dtype=bool)
    for zone, values in comp.zone_scores.items():
        total += float(weights.get(zone, 0.0)) * values
        eligible |= values > 0
    total += beta * authority * eligible
    return [doc_ids[i] for i in top_k_ids(total, eligible, k)]


def linear_ranker(components: Mapping[str, TopicComponents], authority: np.ndarray,
                  doc_ids: Sequence[str], weights_fn: Callable[[str, dict], dict[str, float]],
                  beta_fn: Callable[[str, dict], float], k: int = 100) -> RankFn:
    """Build a RankFn from per-topic components. weights_fn and beta_fn turn (qid, params)
    into zone weights and the authority weight, which is where each variant differs."""
    def rank(qid: str, params: dict) -> list[str]:
        return linear_ranked(components[qid], weights_fn(qid, params), beta_fn(qid, params),
                             authority, doc_ids, k)
    return rank


def title_abstract_weights(w_title: float) -> dict[str, float]:
    """Title weight w_title against abstract fixed at 1, normalized to sum to 1."""
    return {"title": w_title / (w_title + 1.0), "abstract": 1.0 / (w_title + 1.0)}


def grid_search(grid: Mapping[str, Sequence], rank_fn: RankFn, qrels: Mapping[str, Mapping[str, int]],
                qids: Sequence[str], split: Mapping, metric: str = "nDCG@10",
                valid: Callable[[dict], bool] | None = None) -> tuple[list[GridRow], GridRow]:
    """Evaluate every grid point on the tuning topics. Returns (all rows, best row).

    The best row has the highest mean `metric`; ties go to the earliest grid point, so list
    values from the simplest (e.g. beta = 0) upward. `valid` drops impossible combinations.
    """
    assert_tuning_only(qids, split)
    names = list(grid)
    rows: list[GridRow] = []
    for values in itertools.product(*(grid[n] for n in names)):
        params = dict(zip(names, values))
        if valid is not None and not valid(params):
            continue
        run = {q: rank_fn(q, params) for q in qids}
        per_topic = evaluate(run, qrels, qids)
        n = max(len(per_topic), 1)
        mean = {m: sum(t[m] for t in per_topic.values()) / n for m in ("P@10", "nDCG@10", "Recall@100")}
        rows.append(GridRow(params, mean, per_topic))
    if not rows:
        raise ValueError("grid produced no valid points")
    best = rows[0]
    for row in rows[1:]:
        if row.mean[metric] > best.mean[metric]:
            best = row
    return rows, best


def cv_select(rows: Sequence[GridRow], qids: Sequence[str], folds: int = 5, repeats: int = 10,
              metric: str = "nDCG@10", seed: int = 0) -> dict:
    """Repeated k-fold check of the tuning: pick the best grid point on each training fold and
    score it on the held-out topics. Returns the mean held-out score and how often each grid
    point won, a measure of how stable the chosen parameters are."""
    held_out, wins = [], {}
    for train, test in repeated_kfold(qids, k=folds, repeats=repeats, seed=seed):
        best_i = max(range(len(rows)),
                     key=lambda i: (np.mean([rows[i].per_topic[q][metric] for q in train]), -i))
        held_out.append(float(np.mean([rows[best_i].per_topic[q][metric] for q in test])))
        key = json.dumps(rows[best_i].params, sort_keys=True)
        wins[key] = wins.get(key, 0) + 1
    return {"mean_held_out": float(np.mean(held_out)), "std_held_out": float(np.std(held_out)),
            "choice_counts": dict(sorted(wins.items(), key=lambda kv: -kv[1]))}


def make_adaptive_weights_fn(terms_by_qid: Mapping[str, list[str]], index, global_weights: Mapping[str, float]):
    """weights_fn for V7: per-query zone weights from C's adaptive_weights, cached per (qid, alpha)."""
    from prism.gate import adaptive_weights
    cache: dict[tuple[str, float], dict[str, float]] = {}

    def weights_fn(qid: str, params: dict) -> dict[str, float]:
        key = (qid, float(params["alpha"]))
        if key not in cache:
            cache[key] = adaptive_weights(index, terms_by_qid[qid], dict(global_weights), alpha=key[1])
        return cache[key]
    return weights_fn


def make_champion_fn(index, r: int):
    """A drop-in for Index.champion that uses champion size r, built lazily from the public
    postings API: the top r documents by tf / document length (ties: smaller id), ascending,
    or None when the posting list has at most r documents. Lets V3 be swept over r without
    rebuilding the index (the indexer fixes r at build time)."""
    cache: dict[tuple[str, str], np.ndarray | None] = {}

    def champion(term: str, zone: str):
        key = (term, zone)
        if key not in cache:
            docs, tfs = index.postings(term, zone)
            if len(docs) <= r:
                cache[key] = None
            else:
                lengths = np.maximum(index.lengths(zone)[docs], 1)
                order = np.lexsort((docs, -(tfs / lengths)))[:r]
                cache[key] = np.sort(docs[order]).astype(np.int32)
        return cache[key]
    return champion


def champion_quality_loss(full: Mapping[str, Sequence[str]], champ: Mapping[str, Sequence[str]],
                          k: int = 100) -> dict[str, float]:
    """How much a champion-list run differs from the full-postings run: mean Overlap@k of the
    champion top-k with the full top-k. Compare with the metric loss and the latency saved."""
    qids = sorted(set(full) & set(champ))
    if not qids:
        return {"mean_overlap": 0.0}
    return {"mean_overlap": sum(top_overlap(champ[q], full[q], k) for q in qids) / len(qids)}


def save_tuned(path: str | Path, name: str, best: GridRow, rows: Sequence[GridRow],
               metric: str = "nDCG@10") -> None:
    """Merge one tuned variant into results/tuned.json: the chosen parameters, the tuning-half
    score and the whole grid (for the sensitivity plots)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[name] = {"metric": metric, "best_params": best.params, "best_mean": best.mean,
                  "grid": [{"params": r.params, "mean": r.mean} for r in rows]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
