"""Human-readable per-term score trace for a search result."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from prism.config import VARIANTS
from prism.gate import adaptive_weights, beta_of_q, specificity
from prism.search import _has_citation_counts


def explain(result, pq, index, variant: str = "V2") -> str:
    """Explain term weights, zone totals, authority and final net score."""
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}")
    cfg = VARIANTS[variant]
    doc = index.internal_id(result.doc_id)
    if doc < 0:
        raise KeyError(f"document {result.doc_id!r} is not in this index")
    lines = [f"Document: {result.doc_id} — {result.title}"]

    if cfg.scorer == "bm25":
        total = 0.0
        lengths = index.lengths("all")
        avg_len = index.avg_len("all")
        for term, qtf in Counter(pq.terms).items():
            docs, tfs = index.postings(term, "all")
            at = int(np.searchsorted(docs, doc))
            tf = int(tfs[at]) if at < len(docs) and int(docs[at]) == doc else 0
            df = index.df(term, "all")
            if not tf or not df or avg_len <= 0:
                continue
            idf = math.log1p((index.n_docs - df + 0.5) / (df + 0.5))
            part = idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * lengths[doc] / avg_len))
            total += part
            lines.append(f"all {term}: tf={tf}, df={df}, idf={idf:.6f}, BM25 contribution={part:.8f}")
        lines.extend([f"BM25 total: {total:.8f}", f"Search score: {result.score:.8f}"])
        return "\n".join(lines)

    weights = adaptive_weights(index, pq.terms, cfg.zone_weights) if cfg.adaptive_zones else cfg.zone_weights
    beta = beta_of_q(specificity(index, pq.terms), cfg.beta, cfg.s_lo, cfg.s_hi) if cfg.gate else cfg.beta
    n_docs = index.n_docs
    for zone in cfg.zone_weights:
        q_weights = {}
        for term, tf in Counter(pq.terms).items():
            df = index.df(term, zone)
            if df:
                q_weights[term] = (1.0 + math.log10(tf)) * math.log10(n_docs / df)
        qnorm = math.sqrt(sum(value * value for value in q_weights.values())) or 1.0
        doc_norm = index.doc_norm(doc, zone)
        zone_total = 0.0
        lines.append(f"[{zone}] query weight norm={qnorm:.6f}, document norm={doc_norm:.6f}")
        for term, query_weight in q_weights.items():
            docs, tfs = index.postings(term, zone)
            at = int(np.searchsorted(docs, doc))
            tf = int(tfs[at]) if at < len(docs) and int(docs[at]) == doc else 0
            if not tf or not doc_norm:
                contribution = 0.0
                doc_weight = 0.0
            else:
                doc_weight = 1.0 + math.log10(tf)
                contribution = (query_weight / qnorm) * doc_weight / doc_norm
            zone_total += contribution
            lines.append(
                f"{term}: tf={tf}, df={index.df(term, zone)}, idf={math.log10(n_docs / index.df(term, zone)) if index.df(term, zone) else 0.0:.6f}, "
                f"qweight={query_weight:.6f}, dweight={doc_weight:.6f}, contribution={contribution:.8f}"
            )
        lines.append(f"{zone} cosine={zone_total:.8f}; weight={weights[zone]:.6f}; weighted={weights[zone] * zone_total:.8f}")
    zone_sum = sum(float(weights[z]) * result.zone_scores.get(z, 0.0) for z in weights)
    authority_part = beta * result.authority
    phrase_part = 0.0
    if cfg.phrase_boost > 0 and pq.phrases:
        from prism.boolean_ops import phrase_match
        phrase_part = sum(
            cfg.phrase_boost
            for phrase in pq.phrases
            if doc in phrase_match(index, phrase, "all")
        )
    lines.extend([
        f"Authority g(d)={result.authority:.8f}; beta(q)={beta:.8f}; weighted authority={authority_part:.8f}",
        *(["Authority source: publication-year recency fallback (citation counts are unavailable)."]
          if cfg.authority_mode != "none" and not _has_citation_counts(index) else []),
        f"Net score={zone_sum + authority_part:.8f}; search score={result.score:.8f}",
    ])
    return "\n".join(lines)
