"""The tuning stages wired to the real engine, run on the tiny hand-written index."""
import json

import pytest

from eval.pipeline import (
    TuneContext, authority_array, champion_sweep, tune_adaptive_zones, tune_authority, tune_gate,
    tune_v1, write_specificity,
)
from prism import search as search_module
from prism.index import Index
from prism.indexer import build_index
from tests.test_end_to_end import DOCS, QUERIES

QRELS = {"1": {"p1": 2, "p5": 1}, "2": {"p2": 2}, "3": {"p3": 2, "p1": 1}, "4": {"p4": 2},
         "5": {"p5": 2, "p1": 1}, "6": {"p3": 2}}
SPLIT = {"seed": 42, "tune": ["1", "2", "3", "4"], "report": ["5", "6"]}


@pytest.fixture()
def ctx(tmp_path, monkeypatch):
    build_index(DOCS, tmp_path / "idx")
    index = Index.load(tmp_path / "idx")
    monkeypatch.setattr(search_module, "_INDEX", index)
    return TuneContext(index, QUERIES, QRELS, SPLIT, tmp_path / "res", k=10).prepare()


def test_only_tuning_topics_are_used(ctx):
    assert ctx.tune_qids == ["1", "2", "3", "4"]
    assert set(ctx.components) == {"1", "2", "3", "4"}


def test_stages_run_in_order_and_record_their_choices(ctx):
    ctx.results_dir.mkdir()
    rows, best = tune_v1(ctx)
    assert len(rows) == 5 and best.params["w_title"] in (0.5, 1, 2, 3, 4)
    for name, mode in (("V2", "raw"), ("V5", "cohort")):
        rows, best = tune_authority(ctx, name, mode)
        assert len(rows) == 8 and best.params["beta"] in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    rows, best = tune_gate(ctx)
    assert best.params["s_hi"] > best.params["s_lo"]
    rows, best = tune_adaptive_zones(ctx)
    assert len(rows) == 5
    tuned = json.loads((ctx.results_dir / "tuned.json").read_text())
    assert set(tuned) == {"V1", "V2", "V5", "V6", "V7"}
    assert all(0.0 <= t["best_mean"]["nDCG@10"] <= 1.0 for t in tuned.values())


def test_a_later_stage_needs_the_earlier_one(ctx):
    ctx.results_dir.mkdir()
    with pytest.raises(RuntimeError, match="V1"):
        tune_authority(ctx, "V2", "raw")


def test_tuning_matches_search_for_the_chosen_parameters(ctx):
    """The cached-component ranking must equal what search() returns for the same weights."""
    from eval.tune import linear_ranked, title_abstract_weights
    from prism.config import VARIANTS
    from dataclasses import replace
    ctx.results_dir.mkdir()
    w = 3
    VARIANTS["_t"] = replace(VARIANTS["V2"], zone_weights=title_abstract_weights(w), authority_mode="raw", beta=0.1)
    try:
        for qid in ctx.tune_qids:
            from_search = [h.doc_id for h in search_module.search(QUERIES[qid], k=10, variant="_t")]
            cached = linear_ranked(ctx.components[qid], title_abstract_weights(w), 0.1,
                                   authority_array(ctx.index, "raw"), ctx.doc_ids, 10)
            assert cached == from_search, qid
    finally:
        VARIANTS.pop("_t")


def test_champion_sweep_restores_the_index(ctx):
    ctx.results_dir.mkdir()
    points = champion_sweep(ctx, sizes=[1, 100])
    assert [p["r"] for p in points] == [1, 100]
    assert points[1]["mean_overlap"] == pytest.approx(1.0)        # r above every posting list: no change
    assert 0.0 <= points[0]["mean_overlap"] <= 1.0
    assert "champion" not in ctx.index.__dict__
    assert (ctx.results_dir / "champion_sweep.json").exists() and (ctx.results_dir / "champion_reference.json").exists()


def test_write_specificity(ctx):
    ctx.results_dir.mkdir()
    spec = write_specificity(ctx.index, QUERIES, ctx.results_dir)
    assert set(spec) == set(QUERIES) and all(0.0 <= v <= 1.0 for v in spec.values())
