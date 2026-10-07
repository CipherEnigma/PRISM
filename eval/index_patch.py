"""D: replace the metadata-derived parts of a built index without rebuilding the text index.

Postings, positions, norms, lengths and champion lists come from title and abstract text only, so
they do not depend on metadata.csv or citation counts. What does depend on them are the year, journal,
source, publish month, DOI and PubMed ID fields and the two authority arrays. If an index was built
from a wrong or partial metadata.csv, those parts can be recomputed from the right data and written
back, which is far cheaper than re-indexing 171K papers. The recomputation mirrors prism.indexer so the
result is what a rebuild from the same records would give (tests/test_index_patch.py checks this).
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from prism.indexer import _cohort_rank_from_citations
from prism.schema import Record

METADATA_KEYS = ["years", "journals", "sources", "publish_months", "pubmed_ids", "dois",
                 "authority_raw", "authority_cohort"]
_ALLOWED_MODULES = {"numpy", "numpy.core.multiarray", "numpy._core.multiarray", "numpy.core.numeric",
                    "numpy._core.numeric", "numpy.dtypes", "collections", "builtins", "_codecs"}


class _SafeUnpickler(pickle.Unpickler):
    """Only numpy arrays and plain Python containers may be loaded; anything else is refused,
    so a pickle from someone else cannot run code while we read it."""

    def find_class(self, module: str, name: str):
        if module in _ALLOWED_MODULES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from the index file")


def load_payload(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "index.pkl"
    with path.open("rb") as handle:
        return _SafeUnpickler(handle).load()


def load_citations(path: str | Path) -> dict[str, int]:
    """doc id -> citation count from a JSONL cache. Each line needs an id (`doc_id`, `cord_uid` or
    `_id`) and a count (`citations` or `cited_by_count`); lines without a count are skipped."""
    counts: dict[str, int] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc = obj.get("doc_id") or obj.get("cord_uid") or obj.get("_id")
            value = obj.get("citations", obj.get("cited_by_count"))
            if doc is not None and value is not None:
                counts[str(doc)] = int(value)
    return counts


def metadata_arrays(records: list[Record], citations: Mapping[str, int] | None = None) -> dict:
    """The eight metadata-derived index arrays, built the way prism.indexer builds them."""
    n = len(records)
    years = np.zeros(n, dtype=np.int16)
    journals: list = [None] * n
    sources: list = [None] * n
    months: list = [None] * n
    pubmed_ids: list = [None] * n
    dois: list = [None] * n
    counts = np.zeros(n, dtype=np.float64)
    for i, rec in enumerate(records):
        if rec.year is not None:
            years[i] = int(rec.year)
        journals[i] = rec.journal or None
        sources[i] = rec.source or None
        months[i] = rec.publish_month or None
        pubmed_ids[i] = rec.pubmed_id or None
        dois[i] = rec.doi or None
        value = rec.citations if rec.citations is not None else (citations or {}).get(rec.doc_id)
        if value is not None:
            counts[i] = float(value)
    top = float(counts.max()) if n else 0.0
    raw = np.log1p(counts) / np.log1p(top) if top > 0 else np.zeros(n, dtype=np.float64)
    cohort = np.clip(_cohort_rank_from_citations(counts, months), 0.0, 1.0)
    return {"years": years, "journals": np.asarray(journals, dtype=object),
            "sources": np.asarray(sources, dtype=object), "publish_months": np.asarray(months, dtype=object),
            "pubmed_ids": np.asarray(pubmed_ids, dtype=object), "dois": np.asarray(dois, dtype=object),
            "authority_raw": raw, "authority_cohort": cohort}


def coverage(arrays: Mapping) -> dict[str, float]:
    """Share of papers with each field, for before/after reports."""
    n = max(len(arrays["years"]), 1)
    known = lambda a: float(np.mean([x is not None for x in a]))        # noqa: E731
    return {"year": float((arrays["years"] > 0).sum() / n), "publish_month": known(arrays["publish_months"]),
            "doi": known(arrays["dois"]), "journal": known(arrays["journals"]),
            "pubmed_id": known(arrays["pubmed_ids"]), "citations>0": float((arrays["authority_raw"] > 0).sum() / n)}


def patch_payload(payload: dict, records: Iterable[Record], citations: Mapping[str, int] | None = None) -> dict:
    """A copy of the index payload with the metadata-derived keys recomputed from `records`.

    The records must be the same papers in the same order as the index (checked), because internal
    document ids are positions."""
    records = list(records)
    index_ids = [str(d) for d in payload["doc_ids"].tolist()]
    if index_ids != [r.doc_id for r in records]:
        raise ValueError("records do not match the index: different papers or a different order")
    patched = dict(payload)
    patched.update(metadata_arrays(records, citations))
    return patched


def write_payload(payload: dict, out: str | Path) -> Path:
    """Write index.pkl (pickle protocol 4, as Index.save does) atomically into `out`."""
    out = Path(out)
    target = out / "index.pkl" if (out.is_dir() or not out.suffix) else out
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    os.replace(tmp, target)
    return target
