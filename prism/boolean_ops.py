"""B: hand-written postings operations on sorted numpy arrays, plus phrase and NEAR/k.

Every postings list is a sorted numpy array of unique internal doc ids. The hand-written
merges (intersect, union, difference, intersect_skip) are the reference implementations
shown in the video; the np_* versions give identical output and are used on the full index.
tests/test_boolean_ops.py checks that the two agree.
"""
import math

import numpy as np

ID_DTYPE = np.int32


def _ids(values) -> np.ndarray:
    return np.asarray(values, dtype=ID_DTYPE)


# --- Hand-written reference versions: two-pointer merges, O(len(a) + len(b)) ---

def intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ids in both a and b."""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return _ids(out)


def union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ids in a or b (or both), each once."""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])        # at most one of these two tails is non-empty
    out.extend(b[j:])
    return _ids(out)


def difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ids in a but not in b (a AND NOT b)."""
    i = j = 0
    out = []
    while i < len(a):
        if j == len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return _ids(out)


def intersect_skip(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """intersect with skip pointers every sqrt(n) entries (IIR ch. 2.3).

    When one side is behind, jump ahead by a whole skip as long as that does not pass
    the other side's current id; otherwise step by one as in the plain merge.
    """
    skip_a = max(1, int(math.sqrt(len(a))))
    skip_b = max(1, int(math.sqrt(len(b))))
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            if i % skip_a == 0 and i + skip_a < len(a) and a[i + skip_a] <= b[j]:
                while i % skip_a == 0 and i + skip_a < len(a) and a[i + skip_a] <= b[j]:
                    i += skip_a
            else:
                i += 1
        else:
            if j % skip_b == 0 and j + skip_b < len(b) and b[j + skip_b] <= a[i]:
                while j % skip_b == 0 and j + skip_b < len(b) and b[j + skip_b] <= a[i]:
                    j += skip_b
            else:
                j += 1
    return _ids(out)


# --- numpy versions: same output, used at full index size ---

def np_intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.intersect1d(a, b, assume_unique=True).astype(ID_DTYPE, copy=False)


def np_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.union1d(a, b).astype(ID_DTYPE, copy=False)


def np_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.setdiff1d(a, b, assume_unique=True).astype(ID_DTYPE, copy=False)


# --- Multi-way operations ---

def intersect_many(lists: list[np.ndarray], fast: bool = True,
                   steps: list[int] | None = None) -> np.ndarray:
    """AND of several postings lists, shortest list first, stopping once the result is empty.

    Starting from the rarest term keeps every intermediate result as small as possible.
    fast=False uses the hand-written merge (for demos and tests). If steps is given, the
    result size after each intersection is appended to it (fewer entries if it stopped early).
    """
    if not lists:
        raise ValueError("intersect_many needs at least one list")
    op = np_intersect if fast else intersect
    lists = sorted(lists, key=len)
    result = _ids(lists[0])
    for nxt in lists[1:]:
        if len(result) == 0:
            break
        result = op(result, nxt)
        if steps is not None:
            steps.append(len(result))
    return result


def union_many(lists: list[np.ndarray], fast: bool = True) -> np.ndarray:
    """OR of several postings lists."""
    if not lists:
        return _ids([])
    if fast:
        return np.unique(np.concatenate([_ids(x) for x in lists]))
    result = _ids(lists[0])
    for nxt in lists[1:]:
        result = union(result, nxt)
    return result


# --- Positional operations (need the index's word positions) ---

#
# Each has two versions with identical output. The loop version works doc by doc through
# index.positions() and is the one to read first. The vectorized version needs
# index.positions_flat(term, zone) -> (docs, offsets, flat positions) and handles every doc
# at once by encoding each occurrence as one int64 key, doc * STRIDE + position. Keys sort
# by doc then position, and two occurrences are j apart in the same doc exactly when their
# keys are j apart (positions stay far below STRIDE). On common words it is 20-50x faster.

STRIDE = np.int64(1 << 20)


def _has_flat(index) -> bool:
    return callable(getattr(index, "positions_flat", None))


def _keys(index, term: str, zone: str) -> np.ndarray:
    """Sorted int64 keys doc * STRIDE + position, one per occurrence of term in zone."""
    docs, offsets, flat = index.positions_flat(term, zone)
    counts = np.diff(np.asarray(offsets, dtype=np.int64))
    return np.repeat(np.asarray(docs, dtype=np.int64), counts) * STRIDE + np.asarray(flat, dtype=np.int64)


def _docs_of(keys: np.ndarray) -> np.ndarray:
    return np.unique(keys // STRIDE).astype(ID_DTYPE)


def phrase_match(index, terms: list[str], zone: str) -> np.ndarray:
    """Docs where terms appear at consecutive positions, in order, within zone."""
    if not terms:
        return _ids([])
    if len(terms) == 1:
        return _ids(index.postings(terms[0], zone)[0])
    if _has_flat(index):
        return _phrase_match_vectorized(index, terms, zone)
    return _phrase_match_loop(index, terms, zone)


def _phrase_match_vectorized(index, terms: list[str], zone: str) -> np.ndarray:
    """Keep each start key s of the first term while s + j is a key of term j."""
    starts = _keys(index, terms[0], zone)
    for j, term in enumerate(terms[1:], start=1):
        if len(starts) == 0:
            break
        starts = starts[np.isin(starts + j, _keys(index, term, zone))]
    return _docs_of(starts)


def _phrase_match_loop(index, terms: list[str], zone: str) -> np.ndarray:
    """Candidates are the docs containing every term (intersect_many). In each one, start from
    every position p of the first term and keep only the starts where term j sits at p + j.
    """
    docs = intersect_many([index.postings(t, zone)[0] for t in terms])
    out = []
    for d in docs:
        starts = index.positions(terms[0], zone, d)
        for j, term in enumerate(terms[1:], start=1):
            starts = starts[np.isin(starts + j, index.positions(term, zone, d))]
            if len(starts) == 0:
                break
        if len(starts):
            out.append(d)
    return _ids(out)


def near(index, t1: str, t2: str, k: int, zone: str) -> np.ndarray:
    """Docs where t1 and t2 occur within k positions of each other, in either order.

    0 < |p1 - p2| <= k, so `x NEAR/k x` needs two separate occurrences of x.
    """
    if k < 1:
        raise ValueError(f"NEAR distance must be at least 1, got {k}")
    if _has_flat(index):
        return _near_vectorized(index, t1, t2, k, zone)
    return _near_loop(index, t1, t2, k, zone)


def _near_vectorized(index, t1: str, t2: str, k: int, zone: str) -> np.ndarray:
    """For each key of t1, binary-search the nearest key of t2 strictly before and after it.

    A gap of at most k (< STRIDE) can only happen inside one doc. Strictly before/after
    skips the occurrence itself when t1 == t2.
    """
    k1, k2 = _keys(index, t1, zone), _keys(index, t2, zone)
    if len(k1) == 0 or len(k2) == 0:
        return _ids([])
    after = np.searchsorted(k2, k1, side="right")          # first t2 key > k1
    before = np.searchsorted(k2, k1, side="left") - 1      # last t2 key < k1
    gap_after = np.where(after < len(k2), k2[np.minimum(after, len(k2) - 1)] - k1, k + 1)
    gap_before = np.where(before >= 0, k1 - k2[np.maximum(before, 0)], k + 1)
    hit = np.minimum(gap_after, gap_before) <= k
    return _docs_of(k1[hit])


def _near_loop(index, t1: str, t2: str, k: int, zone: str) -> np.ndarray:
    docs = intersect_many([index.postings(t1, zone)[0], index.postings(t2, zone)[0]])
    out = []
    for d in docs:
        p1 = index.positions(t1, zone, d)
        p2 = index.positions(t2, zone, d)
        gaps = np.abs(p1[:, None] - p2[None, :])     # every pair; abstracts are short
        if np.any((gaps > 0) & (gaps <= k)):
            out.append(d)
    return _ids(out)
