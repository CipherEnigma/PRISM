"""Index read API (A implements; B and C consume). Signatures frozen in hour 0.

Internal doc ids are ints 0..N-1; external ids are strings. Adding methods is allowed,
changing existing signatures is not.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class Index:
    n_docs: int
    zones: list[str]  # ["title", "abstract", "all"]

    def __init__(self, data: dict | None = None):
        self.n_docs = 0
        self.zones = ["title", "abstract", "all"]
        self._doc_ids: np.ndarray = np.array([], dtype=object)
        self._titles: np.ndarray = np.array([], dtype=object)
        self._doc_id_lookup: dict[str, int] = {}
        self._postings: dict[str, dict[str, dict[str, np.ndarray]]] = {zone: {} for zone in self.zones}
        self._doc_norms: dict[str, np.ndarray] = {zone: np.array([], dtype=np.float64) for zone in self.zones}
        self._doc_lens: dict[str, np.ndarray] = {zone: np.array([], dtype=np.int32) for zone in self.zones}
        self._avg_lens: dict[str, float] = {zone: 0.0 for zone in self.zones}
        self._field_arrays: dict[str, np.ndarray] = {}
        self._field_maps: dict[str, dict[str, np.ndarray]] = {}
        self._champions: dict[str, dict[str, np.ndarray]] = {zone: {} for zone in self.zones}
        self._authority_raw: np.ndarray = np.array([], dtype=np.float64)
        self._authority_cohort: np.ndarray = np.array([], dtype=np.float64)
        if data is not None:
            self._load_data(data)

    def _load_data(self, data: dict) -> None:
        self.n_docs = int(data.get("n_docs", 0))
        self.zones = list(data.get("zones", self.zones))
        self._doc_ids = np.asarray(data.get("doc_ids", []), dtype=object)
        self._titles = np.asarray(data.get("titles", ["" for _ in range(self.n_docs)]), dtype=object)
        self._doc_id_lookup = {str(doc_id): idx for idx, doc_id in enumerate(self._doc_ids.tolist())}
        self._postings = data.get("postings", {zone: {} for zone in self.zones})
        self._doc_norms = {zone: np.asarray(data.get("doc_norms", {}).get(zone, np.zeros(self.n_docs, dtype=np.float64)), dtype=np.float64) for zone in self.zones}
        self._doc_lens = {zone: np.asarray(data.get("doc_lens", {}).get(zone, np.zeros(self.n_docs, dtype=np.int32)), dtype=np.int32) for zone in self.zones}
        self._avg_lens = {zone: float(data.get("avg_lens", {}).get(zone, 0.0)) for zone in self.zones}
        self._champions = data.get("champions", {zone: {} for zone in self.zones})
        self._authority_raw = np.asarray(data.get("authority_raw", np.zeros(self.n_docs, dtype=np.float64)), dtype=np.float64)
        self._authority_cohort = np.asarray(data.get("authority_cohort", np.zeros(self.n_docs, dtype=np.float64)), dtype=np.float64)

        self._field_arrays["year"] = np.asarray(data.get("years", np.zeros(self.n_docs, dtype=np.int16)), dtype=np.int16)
        self._field_arrays["journal"] = np.asarray(data.get("journals", [None] * self.n_docs), dtype=object)
        self._field_arrays["source"] = np.asarray(data.get("sources", [None] * self.n_docs), dtype=object)
        self._field_arrays["publish_month"] = np.asarray(data.get("publish_months", [None] * self.n_docs), dtype=object)
        self._field_arrays["pubmed_id"] = np.asarray(data.get("pubmed_ids", [None] * self.n_docs), dtype=object)
        self._field_arrays["doi"] = np.asarray(data.get("dois", [None] * self.n_docs), dtype=object)

        self._field_maps["journal"] = {}
        self._field_maps["source"] = {}
        for idx, value in enumerate(self._field_arrays["journal"]):
            if value is not None:
                key = str(value).lower()
                self._field_maps["journal"].setdefault(key, []).append(idx)
        for idx, value in enumerate(self._field_arrays["source"]):
            if value is not None:
                key = str(value).lower()
                self._field_maps["source"].setdefault(key, []).append(idx)

    @classmethod
    def load(cls, path: str | Path) -> "Index":
        path = Path(path)
        if path.is_dir():
            candidate = path / "index.pkl"
            if not candidate.exists():
                for suffix in (".pkl", ".pickle"):
                    match = next(path.glob(f"*{suffix}"), None)
                    if match is not None:
                        candidate = match
                        break
            if not candidate.exists():
                raise FileNotFoundError(f"No serialized index found in {path!s}")
            path = candidate

        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return cls(payload)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        if target.is_dir() or not target.suffix:
            target = target / "index.pkl"
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "n_docs": self.n_docs,
            "zones": self.zones,
            "doc_ids": self._doc_ids,
            "titles": self._titles,
            "doc_norms": {zone: self._doc_norms.get(zone, np.zeros(self.n_docs, dtype=np.float64)) for zone in self.zones},
            "doc_lens": {zone: self._doc_lens.get(zone, np.zeros(self.n_docs, dtype=np.int32)) for zone in self.zones},
            "avg_lens": {zone: self._avg_lens.get(zone, 0.0) for zone in self.zones},
            "postings": self._postings,
            "champions": self._champions,
            "years": self._field_arrays.get("year", np.zeros(self.n_docs, dtype=np.int16)),
            "journals": self._field_arrays.get("journal", np.array([None] * self.n_docs, dtype=object)),
            "sources": self._field_arrays.get("source", np.array([None] * self.n_docs, dtype=object)),
            "publish_months": self._field_arrays.get("publish_month", np.array([None] * self.n_docs, dtype=object)),
            "pubmed_ids": self._field_arrays.get("pubmed_id", np.array([None] * self.n_docs, dtype=object)),
            "dois": self._field_arrays.get("doi", np.array([None] * self.n_docs, dtype=object)),
            "authority_raw": self._authority_raw,
            "authority_cohort": self._authority_cohort,
        }
        with target.open("wb") as handle:
            pickle.dump(payload, handle, protocol=4)

    def doc_id(self, i: int) -> str:
        """Internal -> external id."""
        if 0 <= i < len(self._doc_ids):
            return str(self._doc_ids[i])
        raise IndexError(f"doc id {i} out of range")

    def internal_id(self, doc_id: str) -> int:
        return int(self._doc_id_lookup.get(str(doc_id), -1))

    def postings(self, term: str, zone: str) -> tuple[np.ndarray, np.ndarray]:
        """(doc ids ascending, tfs); empty arrays if the term is unseen."""
        entry = self._postings.get(zone, {}).get(term)
        if entry is None:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        docs = np.asarray(entry.get("docs", []), dtype=np.int32)
        tfs = np.asarray(entry.get("tfs", []), dtype=np.int32)
        return docs, tfs

    def positions(self, term: str, zone: str, doc: int) -> np.ndarray:
        """Sorted word positions of term in doc."""
        term_data = self._postings.get(zone, {}).get(term)
        if term_data is None:
            return np.array([], dtype=np.int32)
        docs = np.asarray(term_data.get("docs", []), dtype=np.int32)
        offsets = np.asarray(term_data.get("offsets", [0]), dtype=np.int32)
        flat = np.asarray(term_data.get("positions", []), dtype=np.int32)
        if not len(docs):
            return np.array([], dtype=np.int32)
        if len(offsets) == len(docs):
            offsets = np.concatenate(([0], offsets))
        pos_idx = np.searchsorted(docs, int(doc), side="left")
        if pos_idx >= len(docs) or docs[pos_idx] != int(doc):
            return np.array([], dtype=np.int32)
        start = int(offsets[pos_idx])
        end = int(offsets[pos_idx + 1])
        return flat[start:end]

    def df(self, term: str, zone: str) -> int:
        return int(len(self.postings(term, zone)[0]))

    def doc_norm(self, doc: int, zone: str) -> float:
        """Cosine norm of the lnc document vector."""
        if zone not in self._doc_norms:
            raise KeyError(f"unknown zone {zone!r}")
        if doc < 0 or doc >= len(self._doc_norms[zone]):
            raise IndexError(f"doc index {doc} out of range")
        return float(self._doc_norms[zone][doc])

    def doc_len(self, doc: int, zone: str) -> int:
        """Tokens in the zone, for BM25."""
        if zone not in self._doc_lens:
            raise KeyError(f"unknown zone {zone!r}")
        if doc < 0 or doc >= len(self._doc_lens[zone]):
            raise IndexError(f"doc index {doc} out of range")
        return int(self._doc_lens[zone][doc])

    def avg_len(self, zone: str) -> float:
        if zone not in self._avg_lens:
            raise KeyError(f"unknown zone {zone!r}")
        return float(self._avg_lens[zone])

    def field_value(self, doc: int, name: str):
        """year, journal, source or publish_month."""
        if name == "year":
            return int(self._field_arrays.get("year", np.zeros(self.n_docs, dtype=np.int16))[doc])
        if name in {"journal", "source", "publish_month", "pubmed_id", "doi"}:
            arr = self._field_arrays.get(name)
            if arr is None:
                return None
            value = arr[doc]
            return None if value is None else value
        raise KeyError(f"unknown field {name!r}")

    def docs_where(self, name: str, op: str, value) -> np.ndarray:
        """Sorted internal ids; op in =, >=, <=, >, <."""
        if name in {"journal", "source"}:
            if op not in {"=", "=="}:
                raise ValueError(f"unsupported comparison op {op!r} for {name}")
            needle = str(value).casefold()
            arr = self._field_arrays.get(name, np.array([None] * self.n_docs, dtype=object))
            return np.asarray(
                [i for i, item in enumerate(arr) if item is not None and needle in str(item).casefold()],
                dtype=np.int32,
            )

        arr = self._field_arrays.get(name)
        if arr is None:
            return np.array([], dtype=np.int32)

        valid = np.ones(len(arr), dtype=bool)
        if name == "year":
            valid = arr != 0
        elif name == "publish_month":
            valid = np.asarray([item is not None for item in arr], dtype=bool)

        mask = np.zeros(len(arr), dtype=bool)
        valid_ids = np.flatnonzero(valid)
        valid_values = arr[valid_ids]
        if op in {"=", "=="}:
            mask[valid_ids] = valid_values == value
        elif op == ">=":
            mask[valid_ids] = valid_values >= value
        elif op == "<=":
            mask[valid_ids] = valid_values <= value
        elif op == ">":
            mask[valid_ids] = valid_values > value
        elif op == "<":
            mask[valid_ids] = valid_values < value
        else:
            raise ValueError(f"unsupported comparison op {op!r}")
        return np.nonzero(mask)[0].astype(np.int32)

    def champion(self, term: str, zone: str) -> np.ndarray | None:
        """Top-r internal ids ascending, or None if not built (use the full postings)."""
        champions = self._champions.get(zone, {})
        if term not in champions:
            return None
        return np.asarray(champions[term], dtype=np.int32)

    def authority(self, doc: int, mode: str = "raw") -> float:
        """g(d) in [0, 1]; modes "raw", "cohort"."""
        if mode == "raw":
            arr = self._authority_raw
        elif mode == "cohort":
            arr = self._authority_cohort
        else:
            raise ValueError(f"unsupported authority mode {mode!r}")
        if doc < 0 or doc >= len(arr):
            raise IndexError(f"doc index {doc} out of range")
        return float(arr[doc])

    def title(self, doc: int) -> str:
        if doc < 0 or doc >= len(self._titles):
            raise IndexError(f"doc index {doc} out of range")
        return str(self._titles[doc])

    # Convenience methods requested by C (added, not changed).
    def norms(self, zone: str) -> np.ndarray:
        """doc_norm for all documents, aligned with internal ids."""
        if zone not in self._doc_norms:
            raise KeyError(f"unknown zone {zone!r}")
        return self._doc_norms[zone].copy()

    def lengths(self, zone: str) -> np.ndarray:
        """doc_len for all documents, aligned with internal ids."""
        if zone not in self._doc_lens:
            raise KeyError(f"unknown zone {zone!r}")
        return self._doc_lens[zone].copy()
