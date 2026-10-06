"""Shared record types. Frozen in hour 0: change only with all four members agreeing."""
from dataclasses import dataclass


@dataclass
class Record:
    doc_id: str                  # external id, the string used in the qrels
    title: str
    abstract: str
    body: str | None = None      # only if full text is joined later
    year: int | None = None
    journal: str | None = None
    source: str | None = None
    pubmed_id: str | None = None
    doi: str | None = None
    publish_month: str | None = None   # "YYYY-MM", needed for age-normalized authority
    citations: int | None = None       # filled in by C's authority step


@dataclass
class Result:
    doc_id: str                  # external id
    score: float                 # net score
    zone_scores: dict[str, float]    # {"title": ..., "abstract": ...}
    authority: float             # g(d) actually used
    beta: float                  # authority weight used for this query
    matched_terms: list[str]
    title: str
