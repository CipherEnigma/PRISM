"""Index read API (A implements; B and C consume). Signatures frozen in hour 0.

Internal doc ids are ints 0..N-1; external ids are strings. Adding methods is allowed,
changing existing signatures is not.
"""
from pathlib import Path

import numpy as np


class Index:
    n_docs: int
    zones: list[str]                       # ["title", "abstract", "all"]

    @classmethod
    def load(cls, path: str | Path) -> "Index":
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    def doc_id(self, i: int) -> str:
        """Internal -> external id."""
        raise NotImplementedError

    def internal_id(self, doc_id: str) -> int:
        raise NotImplementedError

    def postings(self, term: str, zone: str) -> tuple[np.ndarray, np.ndarray]:
        """(doc ids ascending, tfs); empty arrays if the term is unseen."""
        raise NotImplementedError

    def positions(self, term: str, zone: str, doc: int) -> np.ndarray:
        """Sorted word positions of term in doc."""
        raise NotImplementedError

    def df(self, term: str, zone: str) -> int:
        raise NotImplementedError

    def doc_norm(self, doc: int, zone: str) -> float:
        """Cosine norm of the lnc document vector."""
        raise NotImplementedError

    def doc_len(self, doc: int, zone: str) -> int:
        """Tokens in the zone, for BM25."""
        raise NotImplementedError

    def avg_len(self, zone: str) -> float:
        raise NotImplementedError

    def field_value(self, doc: int, name: str):
        """year, journal, source or publish_month."""
        raise NotImplementedError

    def docs_where(self, name: str, op: str, value) -> np.ndarray:
        """Sorted internal ids; op in =, >=, <=, >, <."""
        raise NotImplementedError

    def champion(self, term: str, zone: str) -> np.ndarray | None:
        """Top-r internal ids ascending, or None if not built (use the full postings)."""
        raise NotImplementedError

    def authority(self, doc: int, mode: str = "raw") -> float:
        """g(d) in [0, 1]; modes "raw", "cohort"."""
        raise NotImplementedError

    def title(self, doc: int) -> str:
        raise NotImplementedError

    # Convenience methods requested by C (added, not changed).
    def norms(self, zone: str) -> np.ndarray:
        """doc_norm for all documents, aligned with internal ids."""
        raise NotImplementedError

    def lengths(self, zone: str) -> np.ndarray:
        """doc_len for all documents, aligned with internal ids."""
        raise NotImplementedError
