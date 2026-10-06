"""A: build_index(records) -> files in index/ (per-zone postings, positions, norms, parametric, champions)."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from prism.analyzer import analyze
from prism.index import Index
from prism.schema import Record


def _text_for_zone(record: Record, zone: str) -> str:
    if zone == "title":
        return record.title or ""
    if zone == "abstract":
        return record.abstract or ""
    if zone == "all":
        return f"{record.title or ''} {record.abstract or ''}".strip()
    raise ValueError(f"unknown zone {zone!r}")


def _tokens_for_zone(record: Record, zone: str) -> list[str]:
    if zone != "all":
        return analyze(_text_for_zone(record, zone))
    title_tokens = analyze(record.title or "")
    abstract_tokens = analyze(record.abstract or "")
    # Keep a positional gap so phrases cannot cross the title/abstract boundary.
    return title_tokens + [""] * 10 + abstract_tokens


def _cohort_rank_from_citations(citations: np.ndarray, publish_months: list[str | None]) -> np.ndarray:
    if len(citations) == 0:
        return np.zeros(0, dtype=np.float64)

    cohort_labels: list[str] = []
    for month in publish_months:
        if month is None:
            cohort_labels.append("__unknown__")
        else:
            cohort_labels.append(month)

    df = pd.DataFrame({"citations": citations.astype(float), "cohort": cohort_labels})
    cohort_values = df["cohort"].copy()
    # merge tiny cohorts into the year bucket if they have fewer than 30 documents
    counts = df["cohort"].value_counts()
    for month, count in counts.items():
        if month == "__unknown__":
            continue
        if count < 30:
            year = month.split("-")[0]
            cohort_values[df["cohort"] == month] = year
    df["cohort"] = cohort_values

    # Use the Min method so ties get the same lowest percentile.
    pct = df.groupby("cohort")["citations"].rank(method="min", pct=True)
    unknown = df["cohort"] == "__unknown__"
    if unknown.any():
        global_pct = df["citations"].rank(method="min", pct=True)
        pct[unknown] = global_pct[unknown]
    return pct.fillna(0.0).to_numpy(dtype=np.float64)


def build_index(records: Iterable[Record], out_dir: str | Path) -> None:
    records = list(records)
    if not records:
        raise ValueError(
            "Cannot build an index from zero records. Check that the BEIR corpus "
            "is present under the data directory and contains valid document IDs."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_docs = len(records)
    doc_ids = [record.doc_id for record in records]
    titles = [record.title or "" for record in records]
    years = np.zeros(n_docs, dtype=np.int16)
    journals = [None] * n_docs
    sources = [None] * n_docs
    months = [None] * n_docs
    pubmed_ids = [None] * n_docs
    dois = [None] * n_docs
    citations = np.zeros(n_docs, dtype=np.float64)

    for idx, record in enumerate(records):
        if record.year is not None:
            years[idx] = int(record.year)
        if record.journal:
            journals[idx] = record.journal
        if record.source:
            sources[idx] = record.source
        if record.publish_month:
            months[idx] = record.publish_month
        if record.pubmed_id:
            pubmed_ids[idx] = record.pubmed_id
        if record.doi:
            dois[idx] = record.doi
        if record.citations is not None:
            citations[idx] = float(record.citations)

    postings_by_zone: dict[str, dict[str, dict[str, np.ndarray]]] = {"title": {}, "abstract": {}, "all": {}}
    doc_lens: dict[str, np.ndarray] = {"title": np.zeros(n_docs, dtype=np.int32), "abstract": np.zeros(n_docs, dtype=np.int32), "all": np.zeros(n_docs, dtype=np.int32)}
    doc_norms: dict[str, np.ndarray] = {"title": np.zeros(n_docs, dtype=np.float64), "abstract": np.zeros(n_docs, dtype=np.float64), "all": np.zeros(n_docs, dtype=np.float64)}

    for doc_idx, record in enumerate(records):
        for zone in ("title", "abstract", "all"):
            tokens = _tokens_for_zone(record, zone)
            real_tokens = [token for token in tokens if token]
            doc_lens[zone][doc_idx] = len(real_tokens)
            counts = Counter(real_tokens)
            doc_norms[zone][doc_idx] = math.sqrt(sum((1.0 + math.log10(tf)) ** 2 for tf in counts.values() if tf > 0))
            positions_by_term: dict[str, list[int]] = defaultdict(list)
            for pos, token in enumerate(tokens):
                if not token:
                    continue
                positions_by_term[token].append(pos)
            for term, positions in positions_by_term.items():
                if term not in postings_by_zone[zone]:
                    postings_by_zone[zone][term] = {"docs": [], "tfs": [], "positions": []}
                entry = postings_by_zone[zone][term]
                entry["docs"].append(doc_idx)
                entry["tfs"].append(len(positions))
                entry["positions"].append(np.asarray(positions, dtype=np.int32))

    champions: dict[str, dict[str, np.ndarray]] = {"title": {}, "abstract": {}, "all": {}}
    for zone in ("title", "abstract", "all"):
        for term, entry in postings_by_zone[zone].items():
            docs = np.asarray(entry["docs"], dtype=np.int32)
            tfs = np.asarray(entry["tfs"], dtype=np.int32)
            if len(docs) <= 500:
                postings_by_zone[zone][term] = {
                    "docs": docs,
                    "tfs": tfs,
                    "offsets": np.asarray([0], dtype=np.int32) if len(docs) == 0 else np.cumsum(np.asarray([len(p) for p in entry["positions"]], dtype=np.int32)),
                    "positions": np.concatenate(entry["positions"]) if entry["positions"] else np.array([], dtype=np.int32),
                }
                continue

            scores = []
            for doc_idx, tf in zip(docs, tfs):
                length = int(doc_lens[zone][doc_idx])
                scores.append((doc_idx, tf / length if length else 0.0))
            best = sorted(scores, key=lambda pair: (-pair[1], pair[0]))[:500]
            champions[zone][term] = np.asarray(sorted(doc_idx for doc_idx, _ in best), dtype=np.int32)
            postings_by_zone[zone][term] = {
                "docs": docs,
                "tfs": tfs,
                "offsets": np.asarray([0], dtype=np.int32) if len(docs) == 0 else np.cumsum(np.asarray([len(p) for p in entry["positions"]], dtype=np.int32)),
                "positions": np.concatenate(entry["positions"]) if entry["positions"] else np.array([], dtype=np.int32),
            }

    # Ensure all terms have a clean offset array with a trailing end.
    for zone in ("title", "abstract", "all"):
        for term, entry in postings_by_zone[zone].items():
            if "offsets" not in entry:
                offsets = np.cumsum(np.asarray([len(p) for p in entry["positions"]], dtype=np.int32))
                entry["offsets"] = np.concatenate(([0], offsets))
            else:
                if len(entry["offsets"]) == len(entry["docs"]):
                    entry["offsets"] = np.concatenate(([0], entry["offsets"]))

    max_citations = float(citations.max())
    authority_raw = np.zeros(n_docs, dtype=np.float64)
    if max_citations > 0:
        authority_raw = np.log1p(citations) / np.log1p(max_citations)

    month_values = list(months)
    cohort_rank = _cohort_rank_from_citations(citations, month_values)
    authority_cohort = np.clip(cohort_rank, 0.0, 1.0)

    data = {
        "n_docs": n_docs,
        "zones": ["title", "abstract", "all"],
        "doc_ids": np.asarray(doc_ids, dtype=object),
        "titles": np.asarray(titles, dtype=object),
        "doc_id_lookup": {doc_id: idx for idx, doc_id in enumerate(doc_ids)},
        "years": years,
        "journals": np.asarray(journals, dtype=object),
        "sources": np.asarray(sources, dtype=object),
        "publish_months": np.asarray(months, dtype=object),
        "pubmed_ids": np.asarray(pubmed_ids, dtype=object),
        "dois": np.asarray(dois, dtype=object),
        "postings": postings_by_zone,
        "doc_lens": doc_lens,
        "doc_norms": doc_norms,
        "avg_lens": {zone: float(np.mean(doc_lens[zone])) if n_docs else 0.0 for zone in ("title", "abstract", "all")},
        "champions": champions,
        "authority_raw": authority_raw,
        "authority_cohort": authority_cohort,
    }

    index = Index(data)
    index.save(out_dir)
