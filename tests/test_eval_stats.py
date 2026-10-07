"""Significance tools checked against values worked out by hand."""
import numpy as np
import pytest

from eval.stats import (
    bootstrap_ci, compare_to_baseline, holm_correct, paired_diffs, repeated_kfold,
    sign_test, spearman, spearman_ci,
)


def test_sign_test_four_wins_one_loss():
    # n = 5, min(wins, losses) = 1: tail = (C(5,0) + C(5,1)) / 32 = 6/32, two-sided p = 12/32.
    r = sign_test([0.1, 0.2, 0.3, 0.4, -0.1])
    assert (r["wins"], r["losses"], r["ties"]) == (4, 1, 0)
    assert r["p"] == pytest.approx(12 / 32)


def test_sign_test_all_wins_and_ties_dropped():
    assert sign_test([1, 1, 1, 1, 1])["p"] == pytest.approx(2 / 32)
    r = sign_test([1, 1, 0, 0])
    assert r["ties"] == 2 and r["p"] == pytest.approx(2 * 0.25)


def test_sign_test_no_signal():
    assert sign_test([0, 0, 0])["p"] == 1.0


def test_holm_adjustment():
    # sorted: a .01, c .03, b .04 with m = 3 -> a: .03, c: max(.03, 2*.03) = .06, b: max(.06, .04) = .06
    adj = holm_correct({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_spearman_perfect_reversed_and_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0         # a constant side has no rank correlation
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)   # monotone, not linear


def test_spearman_ci_brackets_a_clear_relationship():
    x = list(range(20))
    rho, lo, hi = spearman_ci(x, [v * 2 + 1 for v in x], n_boot=200)
    assert rho == pytest.approx(1.0) and hi == pytest.approx(1.0) and lo > 0.5


def test_bootstrap_of_constant_differences_is_degenerate():
    mean, lo, hi = bootstrap_ci([0.1] * 10, n_boot=500)
    assert mean == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)


def test_bootstrap_ci_contains_mean_and_is_seeded():
    d = np.array([0.3, -0.1, 0.2, 0.05, 0.0, 0.4, -0.2, 0.1])
    a, b = bootstrap_ci(d, seed=3), bootstrap_ci(d, seed=3)
    assert a == b and a[1] <= a[0] <= a[2]


def test_paired_diffs_align_by_topic():
    d = paired_diffs({"1": 0.5, "2": 0.4, "3": 0.9}, {"2": 0.1, "1": 0.2})
    assert d.tolist() == pytest.approx([0.3, 0.3])       # topic 3 is missing from b and skipped


def test_compare_to_baseline_adds_holm():
    data = {"V0": {str(i): {"nDCG@10": 0.4} for i in range(8)},
            "V2": {str(i): {"nDCG@10": 0.5} for i in range(8)},
            "V5": {str(i): {"nDCG@10": 0.3} for i in range(8)}}
    rows = compare_to_baseline(data, "V0", "nDCG@10", [str(i) for i in range(8)], n_boot=200)
    assert set(rows) == {"V2", "V5"}
    assert rows["V2"]["wins"] == 8 and rows["V5"]["losses"] == 8
    assert rows["V2"]["mean_diff"] == pytest.approx(0.1)
    assert rows["V2"]["p_holm"] >= rows["V2"]["p"]


def test_repeated_kfold_every_topic_tested_once_per_repeat():
    qids = [str(i) for i in range(10)]
    folds = repeated_kfold(qids, k=5, repeats=3, seed=1)
    assert len(folds) == 15
    for r in range(3):
        tests = [q for _, test in folds[r * 5:(r + 1) * 5] for q in test]
        assert sorted(tests) == sorted(qids)
    for train, test in folds:
        assert not set(train) & set(test) and set(train) | set(test) == set(qids)
