"""D: turn the parameters chosen on the tuning half (results/tuned.json) into variant configs.

The configs are built in memory for evaluation (`run_eval.py --tuned`); prism/config.py is not edited.
Each variant inherits the earlier stages' choices, in the same order they were tuned:
V1 title weight -> V2/V5 beta -> V6 gate (beta_max, s_lo, s_hi) -> V7 alpha. V3 and V4 reuse V2's
beta; V3's champion size comes from the sweep (see choose_champion_size).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from eval.tune import title_abstract_weights
from prism.config import VariantConfig


def tuned_variants(base: Mapping[str, VariantConfig], tuned: Mapping[str, Mapping]) -> dict[str, VariantConfig]:
    """Variants whose tuned parameters exist, built on `base`. Stages that were not tuned are left out
    (and so is anything that depends on them)."""
    out: dict[str, VariantConfig] = {}
    best = {name: t["best_params"] for name, t in tuned.items()}
    if "V1" not in best:
        return out
    weights = title_abstract_weights(best["V1"]["w_title"])
    out["V1"] = replace(base["V1"], zone_weights=weights)
    if "V2" in best:
        out["V2"] = replace(base["V2"], zone_weights=weights, beta=best["V2"]["beta"])
        out["V3"] = replace(base["V3"], zone_weights=weights, beta=best["V2"]["beta"])
        out["V4"] = replace(base["V4"], zone_weights=weights, beta=best["V2"]["beta"])
    if "V5" in best:
        out["V5"] = replace(base["V5"], zone_weights=weights, beta=best["V5"]["beta"])
        if "V6" in best:
            g = best["V6"]
            out["V6"] = replace(base["V6"], zone_weights=weights, beta=g["beta_max"], s_lo=g["s_lo"], s_hi=g["s_hi"])
            if "V7" in best:
                out["V7"] = replace(base["V7"], zone_weights=weights, beta=g["beta_max"], s_lo=g["s_lo"],
                                    s_hi=g["s_hi"], alpha=best["V7"]["alpha"])
    return out


def choose_champion_size(points: Sequence[Mapping], tol: float = 0.01) -> int:
    """The smallest champion size whose nDCG@10 is within `tol` of the best size's: as much speed as
    possible without giving up quality. `points` is the sweep from eval.pipeline.champion_sweep."""
    best = max(p["ndcg"] for p in points)
    return min(p["r"] for p in points if p["ndcg"] >= best - tol)
