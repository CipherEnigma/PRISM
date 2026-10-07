"""Tuning on cached components, checked against a five-document example worked out by hand."""
import numpy as np
import pytest

from eval.tune import (
    GridRow, TopicComponents, assert_tuning_only, champion_quality_loss, cv_select, grid_search,
    linear_ranked, linear_ranker, save_tuned, title_abstract_weights, top_k_ids,
)

DOCS = ["d0", "d1", "d2", "d3", "d4"]
TITLE = np.array([0.5, 0.0, 0.2, 0.0, 0.0])
ABSTRACT = np.array([0.0, 0.4, 0.2, 0.0, 0.1])
AUTH = np.array([0.0, 1.0, 0.0, 1.0, 0.0])        # d3 has authority but matches no query term
COMP = TopicComponents({"title": TITLE, "abstract": ABSTRACT})
W = {"title": 0.5, "abstract": 0.5}


def test_top_k_ids_breaks_ties_by_smaller_id():
    scores = np.array([0.2, 0.9, 0.2, 0.2, 0.5])
    eligible = np.ones(5, dtype=bool)
    assert top_k_ids(scores, eligible, 3).tolist() == [1, 4, 0]
    assert top_k_ids(scores, eligible, 10).tolist() == [1, 4, 0, 2, 3]
    eligible[1] = False
    assert top_k_ids(scores, eligible, 2).tolist() == [4, 0]
    assert top_k_ids(scores, np.zeros(5, dtype=bool), 3).tolist() == []


def test_linear_ranked_without_authority():
    # net = .5*title + .5*abstract: d0 .25, d1 .20, d2 .20, d4 .05; d3 matches nothing
    assert linear_ranked(COMP, W, 0.0, AUTH, DOCS, 10) == ["d0", "d1", "d2", "d4"]


def test_authority_reorders_but_never_adds_a_non_match():
    # beta .1 adds .1 to d1 only (d3 is not eligible): d1 .30, d0 .25, d2 .20, d4 .05
    assert linear_ranked(COMP, W, 0.1, AUTH, DOCS, 10) == ["d1", "d0", "d2", "d4"]


def test_title_abstract_weights_normalize():
    w = title_abstract_weights(3)
    assert w == pytest.approx({"title": 0.75, "abstract": 0.25})


def _setup():
    qrels = {"q1": {"d1": 1}}
    split = {"tune": ["q1"], "report": ["q2"]}
    rank = linear_ranker({"q1": COMP}, AUTH, DOCS, lambda q, p: W, lambda q, p: p["beta"], k=10)
    return qrels, split, rank


def test_grid_search_picks_the_beta_that_ranks_the_relevant_document_first():
    qrels, split, rank = _setup()
    rows, best = grid_search({"beta": [0.0, 0.05, 0.1]}, rank, qrels, ["q1"], split)
    # beta 0: d1 is 2nd -> nDCG 1/log2(3); beta .05: d1 .25 ties d0 .25, smaller id d0 wins -> still 2nd;
    # beta .1: d1 first -> nDCG 1
    assert [round(r.mean["nDCG@10"], 3) for r in rows] == [0.631, 0.631, 1.0]
    assert best.params == {"beta": 0.1}
    assert best.mean["P@10"] == pytest.approx(0.1)


def test_grid_search_tie_goes_to_the_earliest_grid_point():
    qrels, split, rank = _setup()
    _, best = grid_search({"beta": [0.0, 0.01]}, rank, qrels, ["q1"], split)
    assert best.params == {"beta": 0.0}


def test_grid_search_refuses_reporting_topics():
    qrels, split, rank = _setup()
    with pytest.raises(ValueError, match="reporting topics"):
        grid_search({"beta": [0.0]}, rank, qrels, ["q1", "q2"], split)
    with pytest.raises(ValueError):
        assert_tuning_only(["q2"], split)


def test_grid_search_valid_filter():
    qrels, split, rank = _setup()
    rows, _ = grid_search({"beta": [0.0, 0.1], "s": [1, 2, 3]}, rank, qrels, ["q1"], split,
                          valid=lambda p: p["s"] != 2)
    assert len(rows) == 4 and all(r.params["s"] != 2 for r in rows)


def test_cv_select_reports_stable_choice():
    def row(beta, score):
        return GridRow({"beta": beta}, {"nDCG@10": score}, {q: {"nDCG@10": score} for q in "abcd"})
    out = cv_select([row(0.0, 0.3), row(0.1, 0.5)], list("abcd"), folds=2, repeats=3)
    assert out["mean_held_out"] == pytest.approx(0.5)
    assert list(out["choice_counts"]) == ['{"beta": 0.1}'] and out["choice_counts"]['{"beta": 0.1}'] == 6


def test_champion_quality_loss():
    full = {"1": ["a", "b", "c", "d"], "2": ["a", "b"]}
    champ = {"1": ["a", "b", "x", "y"], "2": ["a", "b"]}
    assert champion_quality_loss(full, champ, k=4)["mean_overlap"] == pytest.approx((0.5 + 1.0) / 2)   # topic 1: 2 of 4 returned; topic 2: 2 of 2


def test_save_tuned_merges_variants(tmp_path):
    import json
    row = GridRow({"beta": 0.1}, {"nDCG@10": 0.5})
    path = tmp_path / "tuned.json"
    save_tuned(path, "V2", row, [row])
    save_tuned(path, "V1", row, [row])
    data = json.loads(path.read_text())
    assert set(data) == {"V1", "V2"} and data["V2"]["best_params"] == {"beta": 0.1}
