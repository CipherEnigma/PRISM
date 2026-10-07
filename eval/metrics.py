"""D: P@10, nDCG@10 (graded gains), Recall@100. Relevant means qrels score >= 1."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

RELEVANT_MIN = 1   # fixed for the whole project: relevant means qrels score >= 1

Qrels = Mapping[str, Mapping[str, int]]   # qid -> {doc_id: graded score}
Run = Mapping[str, Sequence[str]]         # qid -> ranked doc_ids, best first


def precision_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 10) -> float:
    """Relevant documents in the top k, divided by k (short lists are not rescaled)."""
    hits = sum(1 for d in ranked[:k] if rels.get(d, 0) >= RELEVANT_MIN)
    return hits / k


def ndcg_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 10) -> float:
    """nDCG with gain = qrels score and discount 1 / log2(rank + 1), ranks starting at 1.

    The ideal ordering is the topic's judged documents sorted by score. Unjudged
    documents count as score 0. TREC-COVID qrels contain a few -1 scores (topics 38 and
    50); gains are clamped at 0 so they count as non-relevant, not as a penalty.
    """
    dcg = sum(max(0, rels.get(d, 0)) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((s for s in rels.values() if s > 0), reverse=True)[:k]
    idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def n_relevant(rels: Mapping[str, int]) -> int:
    return sum(1 for s in rels.values() if s >= RELEVANT_MIN)


def recall_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 100) -> float:
    """Relevant documents in the top k, divided by all relevant documents (0 if there are none)."""
    total = n_relevant(rels)
    if total == 0:
        return 0.0
    return sum(1 for d in ranked[:k] if rels.get(d, 0) >= RELEVANT_MIN) / total


def capped_recall_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 100) -> float:
    """Recall@k with the denominator capped at k: relevant in the top k / min(k, all relevant).

    BEIR reports this "capped Recall@100" for TREC-COVID (Appendix G of the BEIR paper), because every
    topic has more than 100 relevant documents and plain Recall@100 can never exceed k / relevant.
    """
    total = n_relevant(rels)
    if total == 0:
        return 0.0
    return sum(1 for d in ranked[:k] if rels.get(d, 0) >= RELEVANT_MIN) / min(k, total)


def judged_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 10) -> float:
    """Share of the top k that has any qrels judgment (a score of 0 still counts as judged).

    Unjudged documents are scored as non-relevant, so a system that retrieves outside the
    judgment pool is penalized unfairly. This measures how exposed a variant is to that.
    """
    return sum(1 for d in ranked[:k] if d in rels) / k


def condensed_ndcg_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 10) -> float:
    """nDCG@k after removing unjudged documents from the ranking (a robustness check)."""
    return ndcg_at_k([d for d in ranked if d in rels], rels, k)


def topic_metrics(ranked: Sequence[str], rels: Mapping[str, int]) -> dict[str, float]:
    return {
        "P@10": precision_at_k(ranked, rels, 10),
        "nDCG@10": ndcg_at_k(ranked, rels, 10),
        "Recall@100": recall_at_k(ranked, rels, 100),
    }


def evaluate(run: Run, qrels: Qrels, qids: Sequence[str] | None = None) -> dict[str, dict[str, float]]:
    """Per-topic metrics. A topic missing from the run scores 0; topics without qrels are skipped."""
    qids = sorted(qrels) if qids is None else list(qids)
    return {q: topic_metrics(run.get(q, []), qrels[q]) for q in qids if q in qrels}


def evaluate_extended(run: Run, qrels: Qrels, qids: Sequence[str] | None = None) -> dict[str, dict[str, float]]:
    """Per-topic main metrics plus Judged@10, Judged@100 and condensed nDCG@10."""
    qids = sorted(qrels) if qids is None else list(qids)
    out = {}
    for q in qids:
        if q not in qrels:
            continue
        ranked = run.get(q, [])
        out[q] = {
            **topic_metrics(ranked, qrels[q]),
            "CappedRecall@100": capped_recall_at_k(ranked, qrels[q], 100),
            "Judged@10": judged_at_k(ranked, qrels[q], 10),
            "Judged@100": judged_at_k(ranked, qrels[q], 100),
            "cNDCG@10": condensed_ndcg_at_k(ranked, qrels[q], 10),
        }
    return out


def mean_metrics(per_topic: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not per_topic:
        return {"P@10": 0.0, "nDCG@10": 0.0, "Recall@100": 0.0}
    names = next(iter(per_topic.values())).keys()
    return {m: sum(t[m] for t in per_topic.values()) / len(per_topic) for m in names}


def relevance_stats(qrels: Qrels, k: int = 100) -> dict:
    """Relevant counts per topic, the score values that appear, and the Recall@k ceiling.

    The ceiling for a topic is min(1, k / relevant): a system returning only relevant
    documents cannot exceed it.
    """
    counts = {q: n_relevant(r) for q, r in qrels.items()}
    scores = sorted({s for r in qrels.values() for s in r.values()})
    ceilings = {q: min(1.0, k / c) for q, c in counts.items() if c > 0}
    return {
        "relevant_per_topic": counts,
        "score_values": scores,
        "mean_relevant": sum(counts.values()) / len(counts) if counts else 0.0,
        "recall_ceiling_per_topic": ceilings,
        "mean_recall_ceiling": sum(ceilings.values()) / len(ceilings) if ceilings else 0.0,
    }
