"""C-owned ranking smoke tests on a tiny hand-built index."""
import numpy as np
import pytest

import prism.parser
from prism.config import VARIANTS
from prism.authority import recency_scores
from prism.explain import explain
from prism.index import Index
from prism.schema import Result
from prism.search import search, set_index
from tests.fake_analyzer import fake_analyze


def _posting(docs, positions):
    offsets = [0]
    flat = []
    for doc_positions in positions:
        flat.extend(doc_positions)
        offsets.append(len(flat))
    return {
        "docs": np.asarray(docs, dtype=np.int32),
        "tfs": np.asarray([len(p) for p in positions], dtype=np.int32),
        "offsets": np.asarray(offsets, dtype=np.int32),
        "positions": np.asarray(flat, dtype=np.int32),
    }


def _tiny_index():
    postings = {
        "title": {
            "covid": _posting([0, 1], [[0], [0]]),
            "symptoms": _posting([0], [[1]]),
        },
        "abstract": {
            "symptoms": _posting([1, 2], [[0], [0]]),
        },
        "all": {
            "covid": _posting([0, 1], [[0], [0]]),
            "symptoms": _posting([0, 1, 2], [[1], [11], [0]]),
        },
    }
    norms = {
        "title": np.asarray([np.sqrt(2), 1, 0], dtype=float),
        "abstract": np.asarray([0, 1, 1], dtype=float),
        "all": np.asarray([np.sqrt(2), np.sqrt(2), 1], dtype=float),
    }
    lengths = {
        "title": np.asarray([2, 1, 0], dtype=np.int32),
        "abstract": np.asarray([0, 1, 1], dtype=np.int32),
        "all": np.asarray([2, 2, 1], dtype=np.int32),
    }
    return Index({
        "n_docs": 3,
        "zones": ["title", "abstract", "all"],
        "doc_ids": np.asarray(["p0", "p1", "p2"], dtype=object),
        "titles": np.asarray(["COVID symptoms", "COVID paper", "Symptom paper"], dtype=object),
        "postings": postings,
        "doc_norms": norms,
        "doc_lens": lengths,
        "avg_lens": {"title": 1.0, "abstract": 2 / 3, "all": 5 / 3},
        "years": np.asarray([2020, 2021, 2022], dtype=np.int16),
        "authority_raw": np.asarray([0.2, 0.5, 1.0]),
        "authority_cohort": np.asarray([0.25, 0.5, 1.0]),
        "champions": {"title": {}, "abstract": {}, "all": {}},
    })


@pytest.fixture
def tiny_index(monkeypatch):
    monkeypatch.setattr(prism.parser, "analyze", fake_analyze)
    index = _tiny_index()
    set_index(index)
    return index


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_search_ranks_real_documents_for_every_variant(variant, tiny_index):
    results = search("covid symptoms", k=3, variant=variant)
    assert results
    assert len(results) <= 3
    assert all(isinstance(result, Result) for result in results)
    assert all(result.doc_id in {"p0", "p1", "p2"} for result in results)
    assert all(results[i].score >= results[i + 1].score for i in range(len(results) - 1))


def test_phrase_variant_boost_is_explained(tiny_index):
    parsed_query = prism.parser.parse('"covid symptoms"')
    results = search('"covid symptoms"', variant="V4")
    assert [result.doc_id for result in results] == ["p0"]
    text = explain(results[0], parsed_query, tiny_index, variant="V4")
    assert "Phrase boost=0.10000000" in text
    assert "search score=" in text


def test_unknown_variant_raises():
    with pytest.raises(KeyError):
        search("x", variant="nope")


def test_recency_fallback_uses_month_when_available():
    scores = recency_scores([2020, 2020, 2021], ["2020-01", "2020-11", "2021-01"])
    assert scores[0] < scores[1] < scores[2]
    assert recency_scores([2020, None], [None, None]).tolist() == [1.0, 0.0]
