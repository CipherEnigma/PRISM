"""search() orchestrator (C). Entry point called by D and the demo.

STUB: returns fake results so D can wire the harness on day one. Replace with the real
pipeline (parse -> candidates -> zoned cosine + authority -> heap top-K) by Gate 2.
"""
from prism.config import VARIANTS
from prism.schema import Result


def search(query: str, k: int = 10, variant: str = "V2") -> list[Result]:
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    return [
        Result(
            doc_id=f"stub{i:03d}",
            score=1.0 / (i + 1),
            zone_scores={"title": 0.0, "abstract": 0.0},
            authority=0.0,
            beta=0.0,
            matched_terms=[],
            title=f"Stub result {i} for {query!r}",
        )
        for i in range(k)
    ]
