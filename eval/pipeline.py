"""D: the tuning stages wired to the real engine (index + search). Thin script: scripts/run_tune.py.

Stages run on the tuning half only and in the spec's order: V1 title weight, V2 beta, V5 beta
(cohort authority), V6 gate (beta_max, s_lo, s_hi), V7 alpha, then the V3 champion-size sweep.
Each stage reads the earlier stages' choices from results/tuned.json and merges its own back in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from eval.runner import latency_summary, run_variant
from eval.tune import (
    GridRow, champion_quality_loss, collect_components, grid_search, linear_ranker,
    make_adaptive_weights_fn, make_champion_fn, save_tuned, title_abstract_weights,
)
from prism.config import VARIANTS

# The spec's grid stops at 0.2, but beta kept improving there on TREC-COVID, so it is extended;
# run_tune.py warns whenever a best value sits on the edge of its grid.
BETAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
TITLE_WEIGHTS = [0.5, 1, 2, 3, 4]
BETA_MAX = [0.05, 0.1, 0.2, 0.4]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CHAMPION_SIZES = [200, 500, 1000, 2000]


def authority_array(index, mode: str) -> np.ndarray:
    """g(d) for every document, with the same recency fallback as prism.search when the index
    holds no citation counts (the array is all zeros)."""
    arr = np.array([index.authority(d, mode) for d in range(index.n_docs)], dtype=np.float64)
    if not np.any(arr):
        from prism.authority import recency_scores
        arr = recency_scores([index.field_value(d, "year") for d in range(index.n_docs)])
    return arr


@dataclass
class TuneContext:
    index: object
    queries: Mapping[str, str]
    qrels: Mapping[str, Mapping[str, int]]
    split: Mapping
    results_dir: Path
    k: int = 100
    components: dict = field(default_factory=dict)
    doc_ids: list = field(default_factory=list)
    terms: dict = field(default_factory=dict)

    @property
    def tune_qids(self) -> list[str]:
        return sorted(q for q in self.split["tune"] if q in self.queries and q in self.qrels)

    def prepare(self) -> "TuneContext":
        """Compute every tuning topic's per-zone cosine arrays and specificity once."""
        from prism.gate import specificity
        from prism.parser import parse
        from prism.search import score_components, set_index
        set_index(self.index)
        self.doc_ids = [self.index.doc_id(i) for i in range(self.index.n_docs)]
        self.terms = {q: parse(self.queries[q]).terms for q in self.tune_qids}
        self.components = collect_components(
            self.queries, self.tune_qids,
            zone_fn=lambda text: score_components(text)[0],
            specificity_fn=lambda text: specificity(self.index, parse(text).terms))
        return self

    def tuned(self) -> dict:
        path = self.results_dir / "tuned.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def best(self, name: str) -> dict:
        t = self.tuned()
        if name not in t:
            raise RuntimeError(f"stage {name} has not been tuned yet; run it first")
        return t[name]["best_params"]

    def _run(self, name: str, grid, authority, weights_fn, beta_fn, valid=None) -> tuple[list[GridRow], GridRow]:
        rank = linear_ranker(self.components, authority, self.doc_ids, weights_fn, beta_fn, self.k)
        rows, best = grid_search(grid, rank, self.qrels, self.tune_qids, self.split, valid=valid)
        save_tuned(self.results_dir / "tuned.json", name, best, rows)
        return rows, best


def tune_v1(ctx: TuneContext):
    zero = np.zeros(ctx.index.n_docs)
    return ctx._run("V1", {"w_title": TITLE_WEIGHTS}, zero,
                    lambda q, p: title_abstract_weights(p["w_title"]), lambda q, p: 0.0)


def tune_authority(ctx: TuneContext, name: str, mode: str):
    """V2 (raw authority) or V5 (cohort authority): the fixed authority weight beta."""
    w = title_abstract_weights(ctx.best("V1")["w_title"])
    return ctx._run(name, {"beta": BETAS}, authority_array(ctx.index, mode),
                    lambda q, p: w, lambda q, p: p["beta"])


def spec_grid(ctx: TuneContext) -> dict:
    """s_lo and s_hi chosen from the spread of the tuning topics' specificity (quantiles)."""
    s = np.array([c.specificity for c in ctx.components.values()])
    lo = sorted({round(float(np.quantile(s, q)), 3) for q in (0.1, 0.25)})
    hi = sorted({round(float(np.quantile(s, q)), 3) for q in (0.75, 0.9)})
    return {"beta_max": BETA_MAX, "s_lo": lo, "s_hi": hi}


def tune_gate(ctx: TuneContext):
    from prism.gate import beta_of_q
    w = title_abstract_weights(ctx.best("V1")["w_title"])
    return ctx._run("V6", spec_grid(ctx), authority_array(ctx.index, "cohort"),
                    lambda q, p: w,
                    lambda q, p: beta_of_q(ctx.components[q].specificity, p["beta_max"], p["s_lo"], p["s_hi"]),
                    valid=lambda p: p["s_hi"] > p["s_lo"])


def tune_adaptive_zones(ctx: TuneContext):
    from prism.gate import beta_of_q
    g = ctx.best("V1")["w_title"]
    gate = ctx.best("V6")
    weights_fn = make_adaptive_weights_fn(ctx.terms, ctx.index, title_abstract_weights(g))
    return ctx._run("V7", {"alpha": ALPHAS}, authority_array(ctx.index, "cohort"), weights_fn,
                    lambda q, p: beta_of_q(ctx.components[q].specificity, gate["beta_max"], gate["s_lo"], gate["s_hi"]))


def champion_sweep(ctx: TuneContext, sizes=CHAMPION_SIZES) -> list[dict]:
    """V3 over champion sizes on the tuning topics: quality, latency and overlap with the full run.

    Index.champion is swapped for an emulation at each r (see make_champion_fn) and restored.
    The reference row (r = null in champion_reference.json) is V3's config with champions off,
    i.e. the same system on full postings (V2's text scoring and authority).
    """
    from prism.search import search
    from eval.metrics import evaluate, mean_metrics
    queries = {q: ctx.queries[q] for q in ctx.tune_qids}

    def run_champions(enabled: bool, r: int | None):
        if r is not None:
            ctx.index.champion = make_champion_fn(ctx.index, r)
        try:
            from dataclasses import replace
            VARIANTS["_sweep"] = replace(VARIANTS["V3"], champions=enabled)
            return run_variant("_sweep", queries, k=ctx.k, search_fn=search)
        finally:
            VARIANTS.pop("_sweep", None)
            ctx.index.__dict__.pop("champion", None)

    full = run_champions(False, None)
    ref = {"ndcg": mean_metrics(evaluate(full.ranked(), ctx.qrels, ctx.tune_qids))["nDCG@10"],
           **latency_summary(full.latency)}
    (ctx.results_dir / "champion_reference.json").write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    points = []
    for r in sizes:
        res = run_champions(True, r)
        m = mean_metrics(evaluate(res.ranked(), ctx.qrels, ctx.tune_qids))
        points.append({"r": r, "ndcg": m["nDCG@10"], "recall100": m["Recall@100"],
                       "median_ms": latency_summary(res.latency)["median_ms"],
                       **champion_quality_loss(full.ranked(), res.ranked(), k=ctx.k)})
    (ctx.results_dir / "champion_sweep.json").write_text(json.dumps(points, indent=2) + "\n", encoding="utf-8")
    return points


def write_specificity(index, queries: Mapping[str, str], results_dir: Path) -> dict[str, float]:
    """results/specificity.json for every topic (the N2 histogram and scatter read it)."""
    from prism.gate import specificity
    from prism.parser import parse
    spec = {q: specificity(index, parse(text).terms) for q, text in sorted(queries.items())}
    (results_dir / "specificity.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec
