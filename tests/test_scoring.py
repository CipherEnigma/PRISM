"""Scoring, gate and authority: hand calculations, a rank_bm25 cross-check, and monotonicity properties.

Three-document corpus (titles empty, so the all zone equals the abstract; N = 3):
  d0  "alpha beta"              -> alpha 1, beta 1                length 2
  d1  "alpha alpha gamma"       -> alpha 2, gamma 1               length 3
  d2  "beta gamma gamma gamma"  -> beta 1, gamma 3                length 4
Every term has df = 2, so idf = log10(3 / 2).
"""
import math
import random

import numpy as np
import pytest

from prism.authority import authority_scores, recency_scores
from prism.gate import adaptive_weights, beta_of_q, specificity
from prism.index import Index
from prism.indexer import build_index
from prism.schema import Record
from prism.scoring import bm25_scores, cosine_scores


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp("scoring")
    build_index([Record("d0", "", "alpha beta"), Record("d1", "", "alpha alpha gamma"),
                 Record("d2", "", "beta gamma gamma gamma")], out)
    return Index.load(out)


def test_cosine_matches_a_hand_calculation(tiny):
    # Query "alpha beta": each term has query weight idf, and after normalizing both are 1/sqrt(2).
    # d0: doc weights (1, 1), norm sqrt(2)                  -> (1/sqrt2)(1 + 1) / sqrt2 = 1.0
    # d1: alpha 1+log10(2) = 1.30103, gamma 1, norm sqrt(1.30103^2 + 1) = 1.64094
    #                                                        -> (1/sqrt2)(1.30103) / 1.64094 = 0.56063
    # d2: beta 1, gamma 1+log10(3) = 1.47712, norm sqrt(1 + 1.47712^2) = 1.78378
    #                                                        -> (1/sqrt2)(1) / 1.78378 = 0.39640
    scores = cosine_scores(tiny, "all", ["alpha", "beta"])
    a = 1 + math.log10(2)
    g = 1 + math.log10(3)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx((1 / math.sqrt(2)) * a / math.sqrt(a * a + 1))
    assert scores[2] == pytest.approx((1 / math.sqrt(2)) * 1 / math.sqrt(1 + g * g))
    assert scores[1] == pytest.approx(0.56063, abs=1e-4) and scores[2] == pytest.approx(0.39640, abs=1e-4)
    assert np.argsort(-scores).tolist() == [0, 1, 2]


def test_cosine_candidates_unseen_terms_and_repeated_query_terms(tiny):
    restricted = cosine_scores(tiny, "all", ["alpha", "beta"], candidates=np.array([1, 2]))
    assert restricted[0] == 0.0 and restricted[1] > 0 and restricted[2] > 0
    assert cosine_scores(tiny, "all", ["zebra"]).tolist() == [0.0, 0.0, 0.0]
    once = cosine_scores(tiny, "all", ["alpha"])
    assert once[0] > 0 and once[2] == 0.0                              # d2 has no alpha
    twice = cosine_scores(tiny, "all", ["alpha", "alpha"])             # query tf 2 scales every weight equally
    np.testing.assert_allclose(twice, once)                           # ... and cosine normalization cancels it


def test_cosine_scores_lie_in_zero_one(tiny):
    for terms in (["alpha"], ["alpha", "beta"], ["alpha", "beta", "gamma"], ["gamma", "gamma"]):
        s = cosine_scores(tiny, "all", terms)
        assert np.all(s >= 0) and np.all(s <= 1 + 1e-12)


def test_bm25_matches_a_hand_calculation(tiny):
    # Query "alpha": df = 2, N = 3 -> idf = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6) = 0.470004; avgdl = 3.
    # d0: tf 1, |d| 2: denom = 1 + 1.2 (0.25 + 0.75 * 2/3) = 1.9  -> 0.470004 * 1 * 2.2 / 1.9 = 0.544215
    # d1: tf 2, |d| 3: denom = 2 + 1.2 (0.25 + 0.75)       = 3.2  -> 0.470004 * 2 * 2.2 / 3.2 = 0.646256
    idf = math.log(1.6)
    scores = bm25_scores(tiny, ["alpha"])
    assert scores[0] == pytest.approx(idf * 2.2 / 1.9)
    assert scores[1] == pytest.approx(idf * 2 * 2.2 / 3.2)
    assert scores[2] == 0.0
    assert scores[0] == pytest.approx(0.544215, abs=1e-5) and scores[1] == pytest.approx(0.646256, abs=1e-5)


def test_bm25_parameters_are_validated(tiny):
    with pytest.raises(ValueError):
        bm25_scores(tiny, ["alpha"], k1=0)
    with pytest.raises(ValueError):
        bm25_scores(tiny, ["alpha"], b=1.5)


def _random_corpus(n_docs=400, vocab=300, seed=7):
    rng = random.Random(seed)
    words = [f"w{i}x" for i in range(vocab)]
    weights = [1.0 / (i + 1) for i in range(vocab)]                      # a Zipf-like word distribution
    return [Record(f"p{i}", "", " ".join(rng.choices(words, weights, k=rng.randint(15, 60)))) for i in range(n_docs)]


def test_bm25_top_ten_overlaps_rank_bm25(tmp_path):
    """Bug check against rank_bm25. Its idf variant differs slightly, so expect close, not identical."""
    rank_bm25 = pytest.importorskip("rank_bm25")
    from prism.analyzer import analyze
    records = _random_corpus()
    build_index(records, tmp_path)
    index = Index.load(tmp_path)
    reference = rank_bm25.BM25Okapi([analyze(r.abstract) for r in records], k1=1.2, b=0.75)
    rng = random.Random(3)
    overlaps = []
    for _ in range(25):
        terms = [t for t in dict.fromkeys(analyze(" ".join(rng.choices([f"w{i}x" for i in range(5, 120)], k=3))))
                 if 0 < index.df(t, "all") < index.n_docs / 2]           # terms with a positive Okapi idf
        if len(terms) < 2:
            continue
        ours = np.argsort(-bm25_scores(index, terms), kind="stable")[:10]
        theirs = np.argsort(-reference.get_scores(terms), kind="stable")[:10]
        overlaps.append(len(set(ours.tolist()) & set(theirs.tolist())) / 10)
    assert len(overlaps) >= 15
    assert np.mean(overlaps) >= 0.8, f"mean top-10 overlap with rank_bm25 only {np.mean(overlaps):.2f}"


def test_specificity_is_in_unit_range_and_higher_for_rarer_terms(tiny):
    assert 0.0 <= specificity(tiny, ["alpha"]) <= 1.0
    assert specificity(tiny, ["zebra"]) == 0.0                           # nothing known, no information
    assert specificity(tiny, []) == 0.0


def test_specificity_rises_with_rarity(tmp_path):
    records = [Record(f"p{i}", "", "common " + ("rare" if i == 0 else "")) for i in range(50)]
    build_index(records, tmp_path)
    ix = Index.load(tmp_path)
    assert specificity(ix, ["rare"]) > specificity(ix, ["common"])
    assert specificity(ix, ["common"]) == pytest.approx(0.0)             # in every document: idf 0


def test_gate_is_monotone_and_bounded():
    """A more specific query never gets a larger beta; beta stays in [0, beta_max]."""
    for beta_max, lo, hi in ((0.2, 0.2, 0.8), (0.05, 0.1, 0.4), (1.0, 0.0, 1.0)):
        grid = [i / 100 for i in range(101)]
        betas = [beta_of_q(s, beta_max, lo, hi) for s in grid]
        assert all(b2 <= b1 + 1e-12 for b1, b2 in zip(betas, betas[1:]))
        assert min(betas) >= 0.0 and max(betas) <= beta_max + 1e-12
        assert beta_of_q(lo, beta_max, lo, hi) == pytest.approx(beta_max)
        assert beta_of_q(hi, beta_max, lo, hi) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        beta_of_q(0.5, 0.1, 0.8, 0.2)
    with pytest.raises(ValueError):
        beta_of_q(0.5, -0.1)


def test_adaptive_weights(tiny, tmp_path):
    glob = {"title": 2 / 3, "abstract": 1 / 3}
    zoned = {"title": 2 / 3, "abstract": 1 / 3}
    assert sum(adaptive_weights(tiny, ["alpha"], zoned, alpha=0.5).values()) == pytest.approx(1.0)
    base = adaptive_weights(tiny, ["alpha"], glob, alpha=0.0)
    assert base == pytest.approx(glob)                                    # alpha 0 keeps the global weights
    assert adaptive_weights(tiny, ["zebra"], glob, alpha=1.0) == pytest.approx(glob)   # no information: unchanged
    with pytest.raises(ValueError):
        adaptive_weights(tiny, ["alpha"], glob, alpha=1.5)


def test_authority_raw_and_cohort_properties():
    months = ["2020-03"] * 40 + ["2020-04"] * 40
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 200, size=80).tolist()
    raw, cohort = authority_scores(counts, months)
    assert np.all((raw >= 0) & (raw <= 1)) and np.all((cohort >= 0) & (cohort <= 1))
    assert raw.max() == pytest.approx(1.0)
    for sl in (slice(0, 40), slice(40, 80)):                              # more citations never lowers the percentile
        order = np.argsort(np.array(counts)[sl], kind="stable")
        assert np.all(np.diff(cohort[sl][order]) >= -1e-12)
    assert np.all(np.diff(raw[np.argsort(counts, kind="stable")]) >= -1e-12)


def test_cohort_percentile_uses_minimum_rank_for_ties():
    """Zero-citation papers all get the lowest percentile, not an average rank."""
    months = ["2020-03"] * 40
    counts = [0] * 30 + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    _, cohort = authority_scores(counts, months)
    assert len(set(cohort[:30].tolist())) == 1 and cohort[0] == pytest.approx(1 / 40)
    assert cohort[-1] == pytest.approx(1.0)


def test_cohort_ignores_age_unlike_raw():
    """The same relative standing in an old and a young cohort gets the same cohort score."""
    months = ["2020-01"] * 40 + ["2020-09"] * 40
    old = list(range(100, 140))                                         # heavily cited because older
    young = list(range(0, 40))                                          # fewer citations, same standing
    raw, cohort = authority_scores(old + young, months)
    assert cohort[:40] == pytest.approx(cohort[40:])
    assert raw[:40].mean() > raw[40:].mean()


def test_recency_scores_increase_with_date_and_unknown_is_zero():
    out = recency_scores([2019, 2020, 2021, None, 0])
    assert out[0] < out[1] < out[2] and out[3] == 0.0 and out[4] == 0.0
    assert recency_scores([2020, 2020], ["2020-02", "2020-11"])[0] < recency_scores([2020, 2020], ["2020-02", "2020-11"])[1]
