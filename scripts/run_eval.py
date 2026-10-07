"""D: python scripts/run_eval.py --variant V0  -> prints P@10, nDCG@10, Recall@100; writes run file + per-topic CSV.

    python scripts/run_eval.py --variant V0          one variant (Gate 1: V0 and R)
    python scripts/run_eval.py --all                 every variant in config.VARIANTS
    python scripts/run_eval.py --report              tables + figures from the saved per-topic CSVs
    python scripts/run_eval.py --all --report        the whole results table, one command
    python scripts/run_eval.py --tuned --all --report   same, with the parameters tuned on the tuning half
                                                      (results/tuned.json), written under runs/tuned and results/tuned
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.data import SPLIT_PATH, load_qrels, load_queries, load_split, select_qids, write_split  # noqa: E402
from eval.metrics import mean_metrics, relevance_stats                                           # noqa: E402
from eval.report import build_report                                                            # noqa: E402
from eval.runner import read_per_topic_csv, run_and_save                                         # noqa: E402
from eval.tune import make_champion_fn                                                           # noqa: E402
from eval.tuned import choose_champion_size, tuned_variants                                      # noqa: E402
from prism.config import RESULTS_DIR, RUNS_DIR, VARIANTS                                         # noqa: E402


def _print_summary(variant: str, summary: dict, per_topic: dict, split: dict, all_qids: list[str]) -> None:
    print(f"\n== {variant}  ({summary['n_topics']} topics, k={summary['k']})")
    for which in ("tune", "report", "all"):
        qids = select_qids(which, split, all_qids)
        m = mean_metrics({q: per_topic[q] for q in qids if q in per_topic})
        print(f"  {which:<7} P@10 {m['P@10']:.4f}   nDCG@10 {m['nDCG@10']:.4f}   Recall@100 {m['Recall@100']:.4f}   Capped {m.get('CappedRecall@100', float('nan')):.4f}"
              f"   Judged@10 {m['Judged@10']:.3f}")
    lat = summary["latency"]
    print(f"  latency  median {lat['median_ms']:.1f} ms   p95 {lat['p95_ms']:.1f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate PRISM variants on the TREC-COVID topics.")
    ap.add_argument("--variant", action="append", help="variant key (repeatable), e.g. V0")
    ap.add_argument("--all", action="store_true", help="run every variant in config.VARIANTS")
    ap.add_argument("--report", action="store_true", help="build tables and figures from saved results")
    ap.add_argument("--tuned", action="store_true", help="use the parameters in results/tuned.json (in memory; config.py is untouched)")
    ap.add_argument("--k", type=int, default=100, help="ranking depth (default 100, needed for Recall@100)")
    ap.add_argument("--data-dir", type=Path, default=None, help="folder with queries.jsonl and qrels/test.tsv")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    variants = sorted(VARIANTS) if args.all else (args.variant or [])
    if not variants and not args.report:
        ap.error("give --variant NAME, --all or --report")
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        ap.error(f"unknown variant(s) {unknown}; choose from {sorted(VARIANTS)}")

    tuned_json = args.results_dir / "tuned.json"
    champion_r = None
    if args.tuned:
        if not tuned_json.exists():
            ap.error(f"--tuned needs {tuned_json}; run scripts/run_tune.py first")
        import json
        VARIANTS.update(tuned_variants(VARIANTS, json.loads(tuned_json.read_text(encoding="utf-8"))))
        sweep = args.results_dir / "champion_sweep.json"
        if sweep.exists():
            champion_r = choose_champion_size(json.loads(sweep.read_text(encoding="utf-8")))
        args.runs_dir, args.results_dir = args.runs_dir / "tuned", args.results_dir / "tuned"
        print(f"tuned parameters applied from {tuned_json}" + (f"; V3 champion size r={champion_r}" if champion_r else ""))

    queries = load_queries(args.data_dir / "queries.jsonl" if args.data_dir else None)
    qrels = load_qrels(args.data_dir / "qrels" / "test.tsv" if args.data_dir else None)
    all_qids = sorted(queries)
    if not SPLIT_PATH.exists():
        write_split(all_qids)
        print(f"Created {SPLIT_PATH} (seeded 25/25 split). Commit it once and never edit it.")
    split = load_split()

    stats = relevance_stats(qrels)
    print(f"qrels score values {stats['score_values']}; mean relevant/topic {stats['mean_relevant']:.1f}; "
          f"Recall@100 ceiling (mean) {stats['mean_recall_ceiling']:.3f}")

    for variant in variants:
        index = None
        if champion_r and variant == "V3":
            from prism.search import _get_index
            index = _get_index()
            index.champion = make_champion_fn(index, champion_r)
        try:
            summary = run_and_save(variant, queries, qrels, k=args.k, runs_dir=args.runs_dir, results_dir=args.results_dir)
        finally:
            if index is not None:
                index.__dict__.pop("champion", None)
        per_topic = read_per_topic_csv(args.results_dir / f"per_topic_{variant}.csv")
        _print_summary(variant, summary, per_topic, split, all_qids)

    if args.report:
        out = build_report(args.results_dir, split, all_qids)
        print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
