"""Turning tuned.json into variant configs, and choosing the champion size."""
import pytest

from eval.report import build_report
from eval.runner import write_per_topic_csv
from eval.tuned import choose_champion_size, tuned_variants
from prism.config import VARIANTS

TUNED = {
    "V1": {"best_params": {"w_title": 0.5}}, "V2": {"best_params": {"beta": 0.3}},
    "V5": {"best_params": {"beta": 0.02}}, "V6": {"best_params": {"beta_max": 0.2, "s_lo": 0.19, "s_hi": 0.3}},
    "V7": {"best_params": {"alpha": 0.25}},
}


def test_every_variant_inherits_the_earlier_stages():
    out = tuned_variants(VARIANTS, TUNED)
    w = {"title": 0.5 / 1.5, "abstract": 1 / 1.5}
    for name in ("V1", "V2", "V3", "V4", "V5", "V6", "V7"):
        assert out[name].zone_weights == pytest.approx(w), name
    assert out["V2"].beta == out["V3"].beta == out["V4"].beta == 0.3
    assert out["V3"].champions and out["V4"].phrase_boost == VARIANTS["V4"].phrase_boost
    assert out["V5"].beta == 0.02 and out["V5"].authority_mode == "cohort"
    assert out["V6"].gate and (out["V6"].beta, out["V6"].s_lo, out["V6"].s_hi) == (0.2, 0.19, 0.3)
    assert out["V7"].adaptive_zones and out["V7"].alpha == 0.25 and out["V7"].s_hi == 0.3


def test_untuned_stages_are_left_out_and_base_is_not_modified():
    before = dict(VARIANTS)
    assert tuned_variants(VARIANTS, {}) == {}
    partial = tuned_variants(VARIANTS, {k: TUNED[k] for k in ("V1", "V2")})
    assert set(partial) == {"V1", "V2", "V3", "V4"}
    assert dict(VARIANTS) == before


def test_champion_size_is_the_smallest_within_tolerance():
    pts = [{"r": 200, "ndcg": 0.465}, {"r": 500, "ndcg": 0.501}, {"r": 1000, "ndcg": 0.531}, {"r": 2000, "ndcg": 0.528}]
    assert choose_champion_size(pts) == 1000
    assert choose_champion_size(pts, tol=0.04) == 500
    assert choose_champion_size([{"r": 200, "ndcg": 0.5}, {"r": 500, "ndcg": 0.505}]) == 200


def test_report_compares_default_and_tuned(tmp_path):
    qids = [str(i) for i in range(1, 9)]
    split = {"seed": 42, "tune": qids[:4], "report": qids[4:]}
    row = lambda v: {"P@10": v, "nDCG@10": v, "Recall@100": v, "CappedRecall@100": v, "Judged@10": 0.9, "Judged@100": 0.9, "cNDCG@10": v}  # noqa: E731
    for d, v in ((tmp_path, 0.40), (tmp_path / "tuned", 0.50)):
        d.mkdir(exist_ok=True)
        for name in ("V0", "V1"):
            write_per_topic_csv(d / f"per_topic_{name}.csv", {q: row(v) for q in qids})
    text = build_report(tmp_path / "tuned", split, qids).read_text()
    assert "## Default against tuned parameters" in text and "| V1 | 0.400 | 0.500 | 0.400 | 0.500 |" in text
