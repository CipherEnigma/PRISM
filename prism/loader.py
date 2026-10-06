"""A: BEIR TREC-COVID + CORD-19 metadata.csv -> Iterator[Record] (join on cord_uid == _id)."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Iterator

from prism.config import ROOT
from prism.schema import Record


def _candidate_data_dirs(data_dir: str | Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if data_dir is not None:
        roots.append(Path(data_dir).expanduser().resolve())
    roots.append(ROOT / "data")
    roots.append(ROOT)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        if root.exists():
            ordered.append(root)
    return ordered


def _find_file(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    for candidate in root.rglob(filename):
        if candidate.is_file():
            return candidate
    return None


def _parse_publish_time(value: object) -> tuple[int | None, str | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if not text or text == "nan":
        return None, None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            from datetime import datetime

            dt = datetime.strptime(text, fmt)
            year = dt.year
            month = f"{dt.year:04d}-{dt.month:02d}" if fmt in {"%Y-%m-%d", "%Y-%m"} else None
            return year, month
        except ValueError:
            continue
    try:
        return int(str(text)[:4]), str(text)[:7]
    except (TypeError, ValueError):
        return None, None


def _metadata_lookup(data_dir: Path) -> dict[str, dict[str, object]]:
    metadata_path = _find_file(data_dir, "metadata.csv")
    if metadata_path is None:
        return {}

    lookup: dict[str, dict[str, object]] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cord_uid = str(row.get("cord_uid") or row.get("Cord UID") or "").strip()
            if not cord_uid or cord_uid in lookup:
                continue
            lookup[cord_uid] = row
    return lookup


def _iter_corpus_documents(data_dir: Path):
    candidates = [
        "corpus.jsonl",
        "corpus.jsonl.gz",
        "corpus.json",
        "collection.jsonl",
        "collection.jsonl.gz",
        "collection.json",
    ]
    for name in candidates:
        path = _find_file(data_dir, name)
        if path is not None:
            yield from _read_corpus_file(path)
            return

    # Try a generic search for files containing a JSONL corpus structure.
    for path in sorted(data_dir.rglob("*.jsonl")):
        if path.name.lower().startswith("qrel") or "metadata" in path.name.lower():
            continue
        yield from _read_corpus_file(path)
        return

    for path in sorted(data_dir.rglob("*.json")):
        if path.name.lower().startswith("qrel") or "metadata" in path.name.lower():
            continue
        yield from _read_corpus_file(path)
        return


def _read_corpus_file(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    if path.suffix == ".jsonl" or path.suffix == ".gz" and path.name.endswith(".jsonl.gz"):
        with opener(path, "rt", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
        return

    with opener(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, dict):
                yield value
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def iter_records(data_dir: str | Path | None = None, metadata_csv: str | Path | None = None) -> Iterator[Record]:
    """Yield records from a BEIR-style corpus joined to CORD-19 metadata when available."""
    searched_dirs = _candidate_data_dirs(data_dir)
    metadata_map: dict[str, dict[str, object]] = {}

    if metadata_csv is not None:
        metadata_path = Path(metadata_csv).expanduser().resolve()
        if metadata_path.exists():
            metadata_map = _metadata_lookup(metadata_path.parent)
    if not metadata_map:
        for root in searched_dirs:
            metadata_map = _metadata_lookup(root)
            if metadata_map:
                break

    corpus_data: list[dict[str, object]] = []
    for root in searched_dirs:
        files = list(_iter_corpus_documents(root))
        if files:
            corpus_data = files
            break

    if not corpus_data:
        searched = ", ".join(str(root) for root in searched_dirs) or "(no existing data directory)"
        raise FileNotFoundError(
            "Could not find a BEIR corpus file (for example corpus.jsonl) in "
            f"the searched locations: {searched}. Download/extract the TREC-COVID "
            "corpus or pass its directory as data_dir."
        )

    for obj in corpus_data:
        doc_id = str(obj.get("_id") or obj.get("id") or obj.get("doc_id") or "").strip()
        if not doc_id:
            continue
        title = str(obj.get("title") or "")
        abstract = str(obj.get("text") or obj.get("abstract") or obj.get("contents") or "")
        metadata = metadata_map.get(doc_id)
        year = None
        journal = None
        source = None
        pubmed_id = None
        doi = None
        publish_month = None
        if metadata:
            year_value = metadata.get("publish_time") or metadata.get("publishTime") or metadata.get("year")
            year, publish_month = _parse_publish_time(year_value)
            journal = str(metadata.get("journal") or "").strip() or None
            source = str(metadata.get("source_x") or metadata.get("source") or "").strip() or None
            pubmed_id = str(metadata.get("pubmed_id") or "").strip() or None
            doi = str(metadata.get("doi") or "").strip() or None

        yield Record(
            doc_id=doc_id,
            title=title,
            abstract=abstract,
            year=year,
            journal=journal,
            source=source,
            pubmed_id=pubmed_id,
            doi=doi,
            publish_month=publish_month,
        )
