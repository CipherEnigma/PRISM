"""D: tune on the tuning half only, in the spec's order.

    python scripts/run_tune.py                       every stage in order
    python scripts/run_tune.py --stage v1 --stage v2 chosen stages (later ones need the earlier ones' results)

Stages: v1 (title weight), v2 (beta, raw authority), v5 (beta, cohort authority), v6 (gate),
v7 (adaptive zones), champions (V3 sweep), specificity (writes results/specificity.json).
Chosen parameters go to results/tuned.json; copy them into prism/config.py (C owns that file).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.data import load_qrels, load_queries, load_split                      # noqa: E402
from eval.pipeline import (                                                      # noqa: E402
    TuneContext, champion_sweep, tune_adaptive_zones, tune_authority, tune_gate, tune_v1,
    write_specificity,
)
from prism.config import INDEX_DIR, RESULTS_DIR                                  # noqa: E402
from prism.index import Index                                                    # noqa: E402

ORDER = ["v1", "v2", "v5", "v6", "v7", "champions", "specificity"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune PRISM variants on the tuning half of the topics.")
    ap.add_argument("--stage", action="append", choices=ORDER, help="stage to run (repeatable); default all")
    ap.add_argument("--index", type=Path, default=INDEX_DIR)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()
    stages = [s for s in ORDER if s in (args.stage or ORDER)]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    queries, qrels, split = load_queries(), load_qrels(), load_split()
    index = Index.load(args.index)
    if stages == ["specificity"]:
        write_specificity(index, queries, args.results_dir)
        print(f"wrote {args.results_dir / 'specificity.json'}")
        return

    ctx = TuneContext(index, queries, qrels, split, args.results_dir).prepare()
    print(f"tuning on {len(ctx.tune_qids)} topics (tuning half only)")
    for stage in stages:
        if stage == "specificity":
            write_specificity(index, queries, args.results_dir)
            print("specificity.json written")
        elif stage == "champions":
            for p in champion_sweep(ctx):
                print(f"  r={p['r']:<5} nDCG@10 {p['ndcg']:.4f}  median {p['median_ms']:.1f} ms  overlap {p['mean_overlap']:.3f}")
        else:
            rows, best = {"v1": lambda: tune_v1(ctx), "v2": lambda: tune_authority(ctx, "V2", "raw"),
                          "v5": lambda: tune_authority(ctx, "V5", "cohort"), "v6": lambda: tune_gate(ctx),
                          "v7": lambda: tune_adaptive_zones(ctx)}[stage]()
            print(f"{stage.upper()}: best {best.params}  nDCG@10 {best.mean['nDCG@10']:.4f}  ({len(rows)} grid points)")
    print(f"\nchosen parameters: {args.results_dir / 'tuned.json'}")


if __name__ == "__main__":
    main()
