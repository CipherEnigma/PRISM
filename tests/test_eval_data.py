"""Loading topics/qrels and the seeded split."""
import json

import pytest

from eval.data import BEIR_DIR, load_qrels, load_queries, load_split, make_split, select_qids, write_split


def test_split_is_seeded_disjoint_and_complete():
    qids = [str(i) for i in range(1, 51)]
    a, b = make_split(qids), make_split(list(reversed(qids)))
    assert a == b                                        # input order does not matter
    assert len(a["tune"]) == len(a["report"]) == 25
    assert not set(a["tune"]) & set(a["report"])
    assert set(a["tune"]) | set(a["report"]) == set(qids)
    assert make_split(qids, seed=1) != a                 # the seed matters


def test_write_split_refuses_to_overwrite(tmp_path):
    path = tmp_path / "split.json"
    write_split(["1", "2", "3", "4"], path)
    first = path.read_text()
    with pytest.raises(FileExistsError):
        write_split(["1", "2", "3", "4"], path)
    assert path.read_text() == first
    assert load_split(path)["seed"] == 42


def test_select_qids():
    split = {"tune": ["1", "2"], "report": ["3", "4"]}
    assert select_qids("tune", split, ["1", "2", "3"]) == ["1", "2"]
    assert select_qids("report", split, ["1", "2", "3"]) == ["3"]
    assert select_qids("all", split, ["3", "1"]) == ["1", "3"]
    with pytest.raises(ValueError):
        select_qids("nope", split, [])


def test_load_queries_and_qrels(tmp_path):
    q = tmp_path / "queries.jsonl"
    q.write_text(json.dumps({"_id": "1", "text": "what is covid", "metadata": {"query": "covid", "narrative": "long"}}) + "\n"
                 + json.dumps({"_id": "2", "text": "masks", "metadata": {"query": "mask", "narrative": "n"}}) + "\n")
    assert load_queries(q) == {"1": "what is covid", "2": "masks"}
    assert load_queries(q, field="query") == {"1": "covid", "2": "mask"}
    t = tmp_path / "test.tsv"
    t.write_text("query-id\tcorpus-id\tscore\n1\tabc\t2\n1\tdef\t0\n2\tabc\t-1\n")
    assert load_qrels(t) == {"1": {"abc": 2, "def": 0}, "2": {"abc": -1}}


@pytest.mark.skipif(not (BEIR_DIR / "queries.jsonl").exists(), reason="TREC-COVID not downloaded")
def test_real_trec_covid_files():
    queries, qrels = load_queries(), load_qrels()
    assert len(queries) == 50 and set(queries) == set(qrels)
    assert sum(len(v) for v in qrels.values()) == 66336
    assert sorted({s for v in qrels.values() for s in v.values()}) == [-1, 0, 1, 2]
