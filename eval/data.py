"""D: load TREC-COVID topics and qrels from the BEIR files, and the seeded tuning/reporting split."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Sequence

from prism.config import DATA_DIR, ROOT, SPLIT_SEED

BEIR_DIR = DATA_DIR / "trec-covid"
SPLIT_PATH = ROOT / "eval" / "split.json"


def load_queries(path: str | Path | None = None, field: str = "text") -> dict[str, str]:
    """Topic id -> query string. BEIR evaluates with the short `text` question; `query` and
    `narrative` (keyword form, long description) live under `metadata`."""
    path = Path(path) if path else BEIR_DIR / "queries.jsonl"
    queries: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            value = obj.get(field) if field == "text" else obj.get("metadata", {}).get(field)
            if value is None:
                raise KeyError(f"topic {obj.get('_id')!r} has no field {field!r}")
            queries[str(obj["_id"])] = str(value)
    return queries


def load_qrels(path: str | Path | None = None) -> dict[str, dict[str, int]]:
    """qid -> {doc_id: graded score}, from BEIR's `query-id corpus-id score` TSV (header row)."""
    path = Path(path) if path else BEIR_DIR / "qrels" / "test.tsv"
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for qid, doc_id, score in reader:
            qrels.setdefault(qid, {})[doc_id] = int(score)
    return qrels


def make_split(qids: Sequence[str], seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    """Seeded 50/50 split of the sorted topic ids into a tuning half and a reporting half."""
    ordered = sorted(qids)
    random.Random(seed).shuffle(ordered)
    half = len(ordered) // 2
    return {"seed": seed, "tune": sorted(ordered[:half]), "report": sorted(ordered[half:])}


def write_split(qids: Sequence[str], path: str | Path = SPLIT_PATH, seed: int = SPLIT_SEED) -> dict:
    """Write eval/split.json. Refuses to overwrite: the split is created once and never changed."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; the split is committed once and never edited")
    split = make_split(qids, seed)
    path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return split


def load_split(path: str | Path = SPLIT_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_qids(which: str, split: dict, all_qids: Sequence[str]) -> list[str]:
    """`tune`, `report` or `all`, as a sorted list."""
    if which == "all":
        return sorted(all_qids)
    if which not in ("tune", "report"):
        raise ValueError(f"unknown topic set {which!r}; choose tune, report or all")
    return sorted(q for q in split[which] if q in set(all_qids))
