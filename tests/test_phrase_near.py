"""B1: phrase_match and near on the five-document corpus, every expectation checked by hand.

Positions are listed under each abstract in tests/fake_index.py. In the all zone the
abstract's positions are shifted by the title length.
"""
import numpy as np
import pytest

from prism.boolean_ops import near, phrase_match
from tests.fake_index import FIVE_DOCS, FakeIndex


@pytest.fixture(scope="module", params=[False, True], ids=["loop", "vectorized"])
def index(request):
    return FakeIndex(FIVE_DOCS, flat=request.param)


@pytest.mark.parametrize("terms, zone, expected", [
    # d0 has contact,trace adjacent. d1 has both but apart (contact 0, trace 2 in the
    # abstract; trace 0, contact 1 in the title). d4 has contact after trace.
    (["contact", "trace"], "abstract", [0]),
    (["contact", "trace"], "title", [0]),
    (["contact", "trace"], "all", [0]),
    # Order matters: d1's title is "trace contact network".
    (["trace", "contact"], "title", [1]),
    (["trace", "contact"], "abstract", []),
    # A repeated word: d4's abstract has trace at 2 and 3.
    (["trace", "trace"], "abstract", [4]),
    # Three-word phrases.
    (["mobile", "contact", "trace"], "abstract", [0]),
    (["contact", "trace", "app"], "all", [0]),
    (["contact", "trace", "mobile"], "abstract", []),
    # One-word phrase is just the postings; unseen words match nothing.
    (["remdesivir"], "abstract", [2, 3]),
    (["contact", "zebra"], "abstract", []),
    ([], "abstract", []),
])
def test_phrase_match(index, terms, zone, expected):
    assert phrase_match(index, terms, zone).tolist() == expected


@pytest.mark.parametrize("t1, t2, k, zone, expected", [
    # remdesivir..trial gaps in the abstract: d2 = 3 (0 -> 3), d3 = 2 (trial 0, remdesivir 2).
    ("remdesivir", "trial", 1, "abstract", []),
    ("remdesivir", "trial", 2, "abstract", [3]),
    ("remdesivir", "trial", 3, "abstract", [2, 3]),
    # Unordered: swapping the terms gives the same answer.
    ("trial", "remdesivir", 2, "abstract", [3]),
    # contact..trace gaps in the abstract: d0 = 1, d1 = 2, d4 = 2 (contact 5, trace 3).
    ("contact", "trace", 1, "abstract", [0]),
    ("contact", "trace", 2, "abstract", [0, 1, 4]),
    # Same word twice needs two occurrences: only d4 has trace twice.
    ("trace", "trace", 1, "abstract", [4]),
    # Zones are separate: in titles only d2 has remdesivir and trial (adjacent).
    ("remdesivir", "trial", 1, "title", [2]),
    ("remdesivir", "zebra", 5, "abstract", []),
])
def test_near(index, t1, t2, k, zone, expected):
    assert near(index, t1, t2, k, zone).tolist() == expected


def test_near_rejects_zero_distance(index):
    with pytest.raises(ValueError):
        near(index, "contact", "trace", 0, "abstract")


def test_fake_index_postings_are_well_formed(index):
    """Same invariants the spec asks of A's real index, so the fake is trustworthy."""
    for zone in index.zones:
        for term in index._pos[zone]:
            ids, tfs = index.postings(term, zone)
            assert (ids[1:] > ids[:-1]).all()
            assert (tfs > 0).all()
            assert index.df(term, zone) == len(ids)
            for d, tf in zip(ids, tfs):
                pos = index.positions(term, zone, d)
                assert len(pos) == tf and (pos[1:] > pos[:-1]).all()


def test_vectorized_matches_loop_on_random_corpus():
    rng = np.random.default_rng(7)
    vocab = ["a1", "b2", "c3", "d4", "e5"]
    docs = [{"title": " ".join(rng.choice(vocab, rng.integers(1, 6))),
             "abstract": " ".join(rng.choice(vocab, rng.integers(1, 40)))} for _ in range(300)]
    loop, vec = FakeIndex(docs), FakeIndex(docs, flat=True)
    for zone in loop.zones:
        for _ in range(40):
            terms = list(rng.choice(vocab, rng.integers(2, 4)))
            assert phrase_match(vec, terms, zone).tolist() == phrase_match(loop, terms, zone).tolist()
            t1, t2, k = *rng.choice(vocab, 2), int(rng.integers(1, 6))
            assert near(vec, t1, t2, k, zone).tolist() == near(loop, t1, t2, k, zone).tolist()
