"""B1: hand-written postings operations agree with numpy and handle edge cases."""
import numpy as np
import pytest

from prism.boolean_ops import (
    difference, intersect, intersect_many, intersect_skip, np_difference, np_intersect,
    np_union, union, union_many,
)


def _random_pairs(n_pairs=1000, seed=0):
    """Sorted unique id arrays of varied sizes and densities, including empty ones."""
    rng = np.random.default_rng(seed)
    for _ in range(n_pairs):
        universe = int(rng.integers(1, 500))
        a = np.sort(rng.choice(universe, size=int(rng.integers(0, universe + 1)), replace=False))
        b = np.sort(rng.choice(universe, size=int(rng.integers(0, universe + 1)), replace=False))
        yield a.astype(np.int32), b.astype(np.int32)


@pytest.mark.parametrize("mine, reference", [
    (intersect, np_intersect),
    (intersect_skip, np_intersect),
    (union, np_union),
    (difference, np_difference),
])
def test_matches_numpy_on_random_pairs(mine, reference):
    for a, b in _random_pairs():
        got, want = mine(a, b), reference(a, b)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, want)


def test_numpy_versions_match_numpy_builtins():
    for a, b in _random_pairs(200, seed=1):
        np.testing.assert_array_equal(np_intersect(a, b), np.intersect1d(a, b))
        np.testing.assert_array_equal(np_union(a, b), np.union1d(a, b))
        np.testing.assert_array_equal(np_difference(a, b), np.setdiff1d(a, b))


A = np.array([1, 3, 5, 7, 9], dtype=np.int32)
B = np.array([2, 3, 4, 9, 10], dtype=np.int32)
EMPTY = np.array([], dtype=np.int32)


def test_small_hand_checked_cases():
    assert intersect(A, B).tolist() == [3, 9]
    assert union(A, B).tolist() == [1, 2, 3, 4, 5, 7, 9, 10]
    assert difference(A, B).tolist() == [1, 5, 7]
    assert difference(B, A).tolist() == [2, 4, 10]


def test_empty_inputs():
    for op in (intersect, intersect_skip, union, difference):
        assert op(EMPTY, EMPTY).tolist() == []
    assert intersect(A, EMPTY).tolist() == []
    assert union(A, EMPTY).tolist() == A.tolist()
    assert difference(A, EMPTY).tolist() == A.tolist()
    assert difference(EMPTY, A).tolist() == []


def test_skip_pointers_on_long_lists():
    a = np.arange(0, 10_000, 3, dtype=np.int32)
    b = np.array([4, 2_999, 3_000, 9_999], dtype=np.int32)
    assert intersect_skip(a, b).tolist() == [3_000, 9_999]
    assert intersect_skip(b, a).tolist() == [3_000, 9_999]


@pytest.mark.parametrize("fast", [True, False])
def test_intersect_many(fast):
    c = np.array([3, 9, 11], dtype=np.int32)
    assert intersect_many([A, B, c], fast=fast).tolist() == [3, 9]
    assert intersect_many([A], fast=fast).tolist() == A.tolist()
    assert intersect_many([A, EMPTY, B], fast=fast).tolist() == []


def test_intersect_many_requires_a_list():
    with pytest.raises(ValueError):
        intersect_many([])


@pytest.mark.parametrize("fast", [True, False])
def test_union_many(fast):
    assert union_many([A, B, EMPTY], fast=fast).tolist() == [1, 2, 3, 4, 5, 7, 9, 10]
    assert union_many([], fast=fast).tolist() == []
