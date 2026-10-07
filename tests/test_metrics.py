"""Metrics checked against toy qrels whose answers are worked out by hand."""
import math

import pytest

from eval.metrics import (
    evaluate, mean_metrics, ndcg_at_k, precision_at_k, recall_at_k, relevance_stats,
)

# Topic judgments: a=2, b=1, c=1 are relevant; x=0 is judged non-relevant; d is unjudged.
RELS = {"a": 2, "b": 1, "c": 1, "x": 0}


def test_precision_counts_relevant_in_top_k():
    ranked = ["a", "x", "b", "d", "c"]
    assert precision_at_k(ranked, RELS, 5) == pytest.approx(3 / 5)
    assert precision_at_k(ranked, RELS, 2) == pytest.approx(1 / 2)


def test_precision_divides_by_k_even_for_short_lists():
    assert precision_at_k(["a"], RELS, 10) == pytest.approx(0.1)


def test_unjudged_and_zero_score_are_not_relevant():
    assert precision_at_k(["x", "d"], RELS, 2) == 0.0


def test_ndcg_perfect_ordering_is_one():
    assert ndcg_at_k(["a", "b", "c"], RELS, 10) == pytest.approx(1.0)


def test_ndcg_hand_computed():
    # ranked: b, a, d.  DCG  = 1/log2(2) + 2/log2(3) + 0 = 1 + 2/log2(3)
    # ideal: a, b, c.   IDCG = 2/log2(2) + 1/log2(3) + 1/log2(4) = 2 + 1/log2(3) + 0.5
    dcg = 1 + 2 / math.log2(3)
    idcg = 2 + 1 / math.log2(3) + 0.5
    assert ndcg_at_k(["b", "a", "d"], RELS, 10) == pytest.approx(dcg / idcg)


def test_ndcg_ideal_is_cut_at_k():
    # With k=1 the ideal is just the score-2 document.
    assert ndcg_at_k(["b"], RELS, 1) == pytest.approx(1 / 2)
    assert ndcg_at_k(["a"], RELS, 1) == pytest.approx(1.0)


def test_negative_qrels_score_is_not_a_penalty():
    # TREC-COVID has a few -1 judgments; retrieving one must score the same as an unjudged doc.
    rels = {"a": 2, "bad": -1}
    assert ndcg_at_k(["bad", "a"], rels, 10) == pytest.approx(ndcg_at_k(["unjudged", "a"], rels, 10))
    assert precision_at_k(["bad"], rels, 1) == 0.0
    assert recall_at_k(["bad"], rels, 100) == 0.0


def test_ndcg_no_relevant_docs_is_zero():
    assert ndcg_at_k(["a"], {"a": 0}, 10) == 0.0


def test_recall_hand_computed():
    ranked = ["a", "d", "b"]
    assert recall_at_k(ranked, RELS, 100) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, RELS, 1) == pytest.approx(1 / 3)


def test_recall_with_no_relevant_is_zero():
    assert recall_at_k(["a"], {"a": 0}, 100) == 0.0


def test_evaluate_and_mean():
    qrels = {"q1": RELS, "q2": {"z": 1}}
    run = {"q1": ["a", "b", "c"]}             # q2 missing from the run scores 0
    per_topic = evaluate(run, qrels)
    assert per_topic["q1"]["Recall@100"] == pytest.approx(1.0)
    assert per_topic["q2"] == {"P@10": 0.0, "nDCG@10": 0.0, "Recall@100": 0.0}
    mean = mean_metrics(per_topic)
    assert mean["Recall@100"] == pytest.approx(0.5)
    assert mean["P@10"] == pytest.approx((0.3 + 0.0) / 2)


def test_evaluate_restricts_to_given_qids():
    qrels = {"q1": RELS, "q2": {"z": 1}}
    assert set(evaluate({}, qrels, ["q2"])) == {"q2"}


def test_relevance_stats_ceiling():
    qrels = {"q1": {f"d{i}": 1 for i in range(400)} | {"n": 0}, "q2": {"a": 2, "b": 1}}
    stats = relevance_stats(qrels, k=100)
    assert stats["relevant_per_topic"] == {"q1": 400, "q2": 2}
    assert stats["score_values"] == [0, 1, 2]
    assert stats["recall_ceiling_per_topic"]["q1"] == pytest.approx(0.25)
    assert stats["recall_ceiling_per_topic"]["q2"] == 1.0


def test_capped_recall_caps_the_denominator_at_k():
    from eval.metrics import capped_recall_at_k
    # 150 relevant documents; the top 100 are all relevant: plain Recall@100 = 100/150, capped = 100/100
    many = {f"d{i}": 1 for i in range(150)}
    ranked = [f"d{i}" for i in range(100)]
    assert recall_at_k(ranked, many, 100) == pytest.approx(100 / 150)
    assert capped_recall_at_k(ranked, many, 100) == pytest.approx(1.0)
    # 4 relevant (fewer than k): the two forms agree; 3 of 4 found
    few = {"a": 2, "b": 1, "c": 1, "d": 1, "x": 0}
    assert capped_recall_at_k(["a", "b", "c", "z"], few, 100) == pytest.approx(3 / 4)
    assert capped_recall_at_k(["a", "b", "c", "z"], few, 100) == recall_at_k(["a", "b", "c", "z"], few, 100)
    assert capped_recall_at_k(["a"], {"x": 0}, 100) == 0.0                       # nothing relevant


def test_evaluate_extended_includes_capped_recall():
    from eval.metrics import evaluate_extended
    many = {f"d{i}": 1 for i in range(150)}
    out = evaluate_extended({"q": [f"d{i}" for i in range(100)]}, {"q": many})
    assert out["q"]["CappedRecall@100"] == pytest.approx(1.0) and out["q"]["Recall@100"] == pytest.approx(100 / 150)
