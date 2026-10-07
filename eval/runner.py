"""D: run_variant(variant, queries, k=100) -> ranked lists; writes runs/<V>.run, results/per_topic_<V>.csv, latency."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from eval.metrics import evaluate_extended, mean_metrics
from prism.config import RESULTS_DIR, RUNS_DIR

SearchFn = Callable[..., list]     # search(query, k, variant) -> objects with .doc_id and .score
PER_TOPIC_COLUMNS = ["qid", "P@10", "nDCG@10", "Recall@100", "Judged@10", "Judged@100", "cNDCG@10", "latency_s"]


@dataclass
class RunResult:
    variant: str
    run: dict[str, list[tuple[str, float]]] = field(default_factory=dict)   # qid -> [(doc_id, score)]
    latency: dict[str, float] = field(default_factory=dict)                 # qid -> seconds

    def ranked(self) -> dict[str, list[str]]:
        return {q: [d for d, _ in docs] for q, docs in self.run.items()}


def default_search() -> SearchFn:
    """The real engine's entry point, imported lazily so the harness works without an index."""
    from prism.search import search
    return search


def run_variant(variant: str, queries: Mapping[str, str], k: int = 100,
                search_fn: SearchFn | None = None, warmup: int = 1) -> RunResult:
    """Run every topic through search(query, k, variant), timing each call.

    `warmup` untimed calls on the first query absorb one-off costs (lazy index load, caches),
    so the latency figures describe steady-state queries.
    """
    search_fn = search_fn or default_search()
    result = RunResult(variant)
    qids = sorted(queries)
    for _ in range(warmup if qids else 0):
        search_fn(queries[qids[0]], k=k, variant=variant)
    for qid in qids:
        start = time.perf_counter()
        hits = search_fn(queries[qid], k=k, variant=variant)
        result.latency[qid] = time.perf_counter() - start
        result.run[qid] = [(h.doc_id, float(h.score)) for h in hits]
    return result


def latency_summary(latency: Mapping[str, float]) -> dict[str, float]:
    """Median and 95th-percentile latency in milliseconds."""
    if not latency:
        return {"median_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    ms = np.array(list(latency.values())) * 1000
    return {"median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)), "mean_ms": float(ms.mean())}


def write_run_file(path: str | Path, run: Mapping[str, Sequence[tuple[str, float]]], tag: str) -> None:
    """TREC format: `qid Q0 docid rank score tag`, rank starting at 1."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for qid in sorted(run, key=lambda q: (len(q), q)):
            for rank, (doc_id, score) in enumerate(run[qid], start=1):
                handle.write(f"{qid} Q0 {doc_id} {rank} {score:.8f} {tag}\n")


def read_run_file(path: str | Path) -> dict[str, list[tuple[str, float]]]:
    run: dict[str, list[tuple[str, float]]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 6:
                run.setdefault(parts[0], []).append((parts[2], float(parts[4])))
    return run


def write_per_topic_csv(path: str | Path, per_topic: Mapping[str, Mapping[str, float]],
                        latency: Mapping[str, float] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PER_TOPIC_COLUMNS)
        for qid in sorted(per_topic, key=lambda q: (len(q), q)):
            row = per_topic[qid]
            writer.writerow([qid] + [f"{row[c]:.6f}" for c in PER_TOPIC_COLUMNS[1:-1]]
                            + [f"{(latency or {}).get(qid, 0.0):.6f}"])


def read_per_topic_csv(path: str | Path) -> dict[str, dict[str, float]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {row["qid"]: {c: float(row[c]) for c in PER_TOPIC_COLUMNS[1:]} for row in csv.DictReader(handle)}


def run_and_save(variant: str, queries: Mapping[str, str], qrels: Mapping[str, Mapping[str, int]],
                 k: int = 100, search_fn: SearchFn | None = None,
                 runs_dir: str | Path = RUNS_DIR, results_dir: str | Path = RESULTS_DIR) -> dict:
    """Run a variant on all given topics and save the run file, per-topic CSV and a summary JSON.

    Per-topic metrics are always computed on every topic, so any subset (tuning half,
    reporting half, all 50) can be reported later from the CSV without re-running search.
    """
    result = run_variant(variant, queries, k=k, search_fn=search_fn)
    per_topic = evaluate_extended(result.ranked(), qrels, sorted(queries))
    write_run_file(Path(runs_dir) / f"{variant}.run", result.run, tag=variant)
    write_per_topic_csv(Path(results_dir) / f"per_topic_{variant}.csv", per_topic, result.latency)
    summary = {"variant": variant, "n_topics": len(per_topic), "k": k,
               "mean": mean_metrics(per_topic), "latency": latency_summary(result.latency)}
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"summary_{variant}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
