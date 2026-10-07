"""Runner, run/CSV files, latency summary and the placeholder baselines."""
import json

import pytest

from eval.baselines import Hit, top_overlap, tfidf_search_fn
from eval.metrics import condensed_ndcg_at_k, judged_at_k
from eval.runner import (
    latency_summary, read_per_topic_csv, read_run_file, run_and_save, run_variant,
    write_per_topic_csv, write_run_file,
)

QUERIES = {"1": "alpha", "2": "beta"}
QRELS = {"1": {"a": 2, "b": 1, "x": 0}, "2": {"c": 1}}
RANKS = {"alpha": ["a", "z", "b"], "beta": ["q", "c"]}


def fake_search(query, k=10, variant="V0"):
    return [Hit(d, 1.0 / (i + 1)) for i, d in enumerate(RANKS[query][:k])]


def test_run_variant_collects_ranked_lists_and_latency():
    calls = []

    def counting(query, k=10, variant="V0"):
        calls.append(query)
        return fake_search(query, k, variant)

    r = run_variant("V0", QUERIES, k=2, search_fn=counting)
    assert r.ranked() == {"1": ["a", "z"], "2": ["q", "c"]}
    assert r.run["1"][0] == ("a", 1.0)
    assert set(r.latency) == {"1", "2"} and all(t >= 0 for t in r.latency.values())
    assert len(calls) == 3                       # one untimed warm-up plus two topics


def test_trec_run_file_round_trip(tmp_path):
    run = {"2": [("c", 0.5)], "10": [("a", 0.9), ("b", 0.4)]}
    path = tmp_path / "runs" / "V0.run"
    write_run_file(path, run, tag="V0")
    lines = path.read_text().splitlines()
    assert lines[0] == "2 Q0 c 1 0.50000000 V0"            # topics sort numerically, rank starts at 1
    assert lines[2] == "10 Q0 b 2 0.40000000 V0"
    assert read_run_file(path) == run


def test_latency_summary_in_milliseconds():
    s = latency_summary({"1": 0.001, "2": 0.002, "3": 0.003})
    assert s["median_ms"] == pytest.approx(2.0) and s["mean_ms"] == pytest.approx(2.0)
    assert s["p95_ms"] == pytest.approx(2.9)
    assert latency_summary({})["median_ms"] == 0.0


def test_judged_and_condensed_metrics_by_hand():
    rels = {"a": 2, "b": 1, "x": 0}
    ranked = ["a", "z", "b", "w"]                      # z and w are unjudged
    assert judged_at_k(ranked, rels, 4) == pytest.approx(0.5)
    assert judged_at_k(["x"], rels, 10) == pytest.approx(0.1)        # a judged 0 still counts as judged
    # condensed ranking is [a, b]: DCG = 2 + 1/log2(3), ideal is the same list
    assert condensed_ndcg_at_k(ranked, rels, 10) == pytest.approx(1.0)


def test_run_and_save_writes_everything(tmp_path):
    s = run_and_save("V0", QUERIES, QRELS, k=100, search_fn=fake_search,
                     runs_dir=tmp_path / "runs", results_dir=tmp_path / "results")
    assert (tmp_path / "runs" / "V0.run").exists()
    per = read_per_topic_csv(tmp_path / "results" / "per_topic_V0.csv")
    # topic 1: a (rel 2) at rank 1, b (rel 1) at rank 3; 2 relevant docs, both retrieved
    assert per["1"]["P@10"] == pytest.approx(0.2)
    assert per["1"]["Recall@100"] == pytest.approx(1.0)
    assert per["1"]["Judged@10"] == pytest.approx(0.2)
    assert per["2"]["P@10"] == pytest.approx(0.1) and per["2"]["Recall@100"] == pytest.approx(1.0)
    assert s["n_topics"] == 2 and s["mean"]["Recall@100"] == pytest.approx(1.0)
    assert json.loads((tmp_path / "results" / "summary_V0.json").read_text())["variant"] == "V0"


def test_per_topic_csv_round_trip(tmp_path):
    rows = {"1": {c: 0.25 for c in ("P@10", "nDCG@10", "Recall@100", "CappedRecall@100", "Judged@10", "Judged@100", "cNDCG@10")}}
    write_per_topic_csv(tmp_path / "t.csv", rows, {"1": 0.012})
    back = read_per_topic_csv(tmp_path / "t.csv")
    assert back["1"]["nDCG@10"] == 0.25 and back["1"]["latency_s"] == pytest.approx(0.012)


def test_top_overlap():
    assert top_overlap(["a", "b", "c", "d"], ["b", "a", "x", "y"], k=4) == pytest.approx(0.5)
    assert top_overlap([], ["a"], k=3) == 0.0
    assert top_overlap([], [], k=3) == 1.0
    assert top_overlap(["a", "b"], ["b", "a", "c"], k=10) == 1.0      # short lists are not penalized


def test_placeholder_tfidf_ranks_the_matching_document_first():
    pytest.importorskip("sklearn")
    docs = {"d1": "remdesivir trial in covid patients", "d2": "mask wearing and transmission",
            "d3": "vaccine efficacy and antibody response"}
    search = tfidf_search_fn(docs)
    assert search("remdesivir trial", k=3)[0].doc_id == "d1"
    assert [h.doc_id for h in search("antibody", k=3)] == ["d3"]     # only matching docs are returned
    assert search("remdesivir trial", k=3) == search("remdesivir trial", k=3)   # deterministic
