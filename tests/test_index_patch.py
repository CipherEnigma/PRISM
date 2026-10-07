"""Patching an index's metadata must equal rebuilding it from the right metadata."""
import dataclasses
import pickle

import numpy as np
import pytest

from eval.index_patch import (
    METADATA_KEYS, coverage, load_citations, load_payload, metadata_arrays, patch_payload, write_payload,
)
from prism.index import Index
from prism.indexer import build_index
from prism.schema import Record

GOOD = [
    Record("a", "Remdesivir trial", "Remdesivir trial in patients.", year=2020, journal="Lancet", source="PMC",
           pubmed_id="111", doi="10.1/a", publish_month="2020-05"),
    Record("b", "Mask wearing", "Masks reduce transmission.", year=2020, journal="BMJ", source="Elsevier",
           doi="10.1/b", publish_month="2020-05"),
    Record("c", "Vaccine efficacy", "Vaccine efficacy in a large trial.", year=2021, journal="NEJM",
           publish_month="2021-03"),
    Record("d", "Old coronavirus study", "A study of coronavirus from long ago.", year=2003),
]
# What a partial metadata.csv produces: papers b and d have no metadata at all.
BAD = [GOOD[0], dataclasses.replace(GOOD[1], year=None, journal=None, source=None, doi=None, publish_month=None),
       GOOD[2], dataclasses.replace(GOOD[3], year=None)]
CITES = {"a": 40, "b": 5, "c": 0}


def _payload(tmp_path, records, name):
    build_index(records, tmp_path / name)
    return load_payload(tmp_path / name)


def _same(x, y):
    np.testing.assert_array_equal(np.asarray(x, dtype=object) if x.dtype == object else x, y)


def test_patching_wrong_metadata_equals_a_rebuild(tmp_path):
    wrong, right = _payload(tmp_path, BAD, "wrong"), _payload(tmp_path, GOOD, "right")
    patched = patch_payload(wrong, GOOD)
    for key in METADATA_KEYS:
        _same(patched[key], right[key])
    assert (wrong["years"] == 0).sum() == 2 and (patched["years"] == 0).sum() == 0


def test_text_index_is_left_untouched(tmp_path):
    wrong = _payload(tmp_path, BAD, "wrong")
    patched = patch_payload(wrong, GOOD)
    for key in set(wrong) - set(METADATA_KEYS):
        assert patched[key] is wrong[key], key            # same objects: nothing was recomputed or copied


def test_citations_added_without_rebuilding_match_a_rebuild(tmp_path):
    with_cites = [dataclasses.replace(r, citations=CITES.get(r.doc_id)) for r in GOOD]
    right = _payload(tmp_path, with_cites, "right")
    patched = patch_payload(_payload(tmp_path, BAD, "wrong"), GOOD, CITES)
    for key in METADATA_KEYS:
        _same(patched[key], right[key])
    assert patched["authority_raw"][0] == pytest.approx(1.0) and patched["authority_raw"][3] == 0.0


def test_mismatched_records_are_refused(tmp_path):
    payload = _payload(tmp_path, GOOD, "right")
    with pytest.raises(ValueError, match="do not match"):
        patch_payload(payload, list(reversed(GOOD)))
    with pytest.raises(ValueError):
        patch_payload(payload, GOOD[:3])


def test_patched_index_loads_and_answers_filters(tmp_path):
    patched = patch_payload(_payload(tmp_path, BAD, "wrong"), GOOD)
    write_payload(patched, tmp_path / "out")
    index = Index.load(tmp_path / "out")
    assert index.docs_where("year", ">=", 2020).tolist() == [0, 1, 2]
    assert index.field_value(1, "journal") == "BMJ"
    assert index.postings("remdesivir", "all")[0].tolist() == [0]       # text index still works


def test_loader_refuses_pickles_that_run_code(tmp_path):
    class Evil:
        def __reduce__(self):
            import os
            return (os.system, ("echo pwned",))
    bad = tmp_path / "index.pkl"
    bad.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        load_payload(bad)


def test_load_citations_accepts_both_key_styles(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"doc_id": "a", "citations": 3}\n{"cord_uid": "b", "cited_by_count": 7}\n{"doc_id": "c"}\n\n')
    assert load_citations(p) == {"a": 3, "b": 7}


def test_coverage_reports_shares():
    cov = coverage(metadata_arrays(GOOD, CITES))
    assert cov["year"] == 1.0 and cov["doi"] == 0.5 and cov["journal"] == 0.75 and cov["citations>0"] == 0.5
