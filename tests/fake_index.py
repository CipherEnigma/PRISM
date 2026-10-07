"""In-memory stand-in for A's Index, used by B's tests until the real index exists.

Documents are given as already-analyzed text (space-separated tokens), so the tests do not
depend on the analyzer. Implements the parts of the Index read API that B uses: postings,
positions, df, field_value, docs_where and champion. The all zone is title then abstract.
"""
import operator

import numpy as np

_OPS = {"=": operator.eq, ">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt}


class FakeIndex:
    zones = ["title", "abstract", "all"]

    def __init__(self, docs: list[dict], champion_r: int | None = None, flat: bool = False):
        """docs: dicts with "title" and "abstract" (analyzed tokens) plus optional fields.

        flat=True also offers positions_flat(), which switches phrase and NEAR to their
        vectorized versions; flat=False exercises the doc-by-doc loops.
        """
        self.docs = docs
        if flat:
            self.positions_flat = self._positions_flat
        self.n_docs = len(docs)
        self.champion_r = champion_r
        # self._pos[zone][term][doc] = sorted positions
        self._pos: dict[str, dict[str, dict[int, np.ndarray]]] = {z: {} for z in self.zones}
        for d, doc in enumerate(docs):
            tokens = {"title": doc["title"].split(), "abstract": doc["abstract"].split()}
            tokens["all"] = tokens["title"] + tokens["abstract"]
            for zone, toks in tokens.items():
                for p, t in enumerate(toks):
                    self._pos[zone].setdefault(t, {}).setdefault(d, []).append(p)
        for zone in self.zones:
            for term, by_doc in self._pos[zone].items():
                for d in by_doc:
                    by_doc[d] = np.array(by_doc[d], dtype=np.int32)

    def doc_id(self, i: int) -> str:
        return f"d{i}"

    def internal_id(self, doc_id: str) -> int:
        return int(doc_id[1:])

    def postings(self, term: str, zone: str) -> tuple[np.ndarray, np.ndarray]:
        by_doc = self._pos[zone].get(term, {})
        ids = sorted(by_doc)
        return (np.array(ids, dtype=np.int32),
                np.array([len(by_doc[d]) for d in ids], dtype=np.int32))

    def positions(self, term: str, zone: str, doc: int) -> np.ndarray:
        return self._pos[zone].get(term, {}).get(doc, np.array([], dtype=np.int32))

    def _positions_flat(self, term: str, zone: str):
        """(docs, offsets with a leading 0, all positions concatenated in doc order)."""
        ids = self.postings(term, zone)[0]
        pos = [self.positions(term, zone, d) for d in ids]
        offsets = np.concatenate(([0], np.cumsum([len(p) for p in pos]))).astype(np.int32)
        flat = np.concatenate(pos).astype(np.int32) if pos else np.array([], dtype=np.int32)
        return ids, offsets, flat

    def df(self, term: str, zone: str) -> int:
        return len(self._pos[zone].get(term, {}))

    def field_value(self, doc: int, name: str):
        return self.docs[doc].get(name)

    def docs_where(self, name: str, op: str, value) -> np.ndarray:
        cmp = _OPS[op]
        ids = [d for d in range(self.n_docs)
               if (v := self.field_value(d, name)) is not None and cmp(v, value)]
        return np.array(ids, dtype=np.int32)

    def champion(self, term: str, zone: str) -> np.ndarray | None:
        """Top-r docs by term frequency (ties by id), ascending; None if not built."""
        if self.champion_r is None:
            return None
        ids, tfs = self.postings(term, zone)
        top = sorted(zip(ids, tfs), key=lambda x: (-x[1], x[0]))[: self.champion_r]
        return np.array(sorted(d for d, _ in top), dtype=np.int32)


# The five-document test corpus from the spec. Positions are written under each abstract
# so the expected phrase and NEAR results in the tests can be checked by eye.
FIVE_DOCS = [
    {   # d0: "contact trace" adjacent in both title and abstract
        "title": "contact trace app",
        #          0      1       2     3   4      5      6
        "abstract": "mobile contact trace app reduce spread covid19",
        "year": 2020, "journal": "lancet", "source": "pmc",
    },
    {   # d1: contact and trace both present but never adjacent
        "title": "trace contact network",
        #          0       1       2     3      4     5
        "abstract": "contact network trace mobile phone data",
        "year": 2021, "journal": "nature", "source": "medline",
    },
    {   # d2: remdesivir and trial 3 apart in the abstract
        "title": "remdesivir trial",
        #          0          1      2       3     4      5
        "abstract": "remdesivir random control trial hospit patient",
        "year": 2020, "journal": "nejm", "source": "pmc",
    },
    {   # d3: trial before remdesivir (unordered NEAR), hydroxychloroquin in title
        "title": "hydroxychloroquin trial",
        #          0      1                 2          3      4      5
        "abstract": "trial hydroxychloroquin remdesivir compar outcom mortal",
        "year": 2021, "journal": "lancet", "source": "medline",
    },
    {   # d4: no year; "trace" appears twice, "contact" once at the end
        "title": "vaccin efficaci",
        #          0      1        2     3     4     5
        "abstract": "vaccin efficaci trace trace mobile contact",
        "journal": "bmj", "source": "arxiv",
    },
]
