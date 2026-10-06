"""A: build_index(records) -> files in index/ (per-zone postings, positions, norms, parametric, champions)."""
from pathlib import Path
from typing import Iterable

from prism.schema import Record


def build_index(records: Iterable[Record], out_dir: str | Path) -> None:
    raise NotImplementedError
