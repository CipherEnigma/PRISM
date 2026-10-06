"""A: BEIR TREC-COVID + CORD-19 metadata.csv -> Iterator[Record] (join on cord_uid == _id)."""
from typing import Iterator

from prism.schema import Record


def iter_records() -> Iterator[Record]:
    raise NotImplementedError
