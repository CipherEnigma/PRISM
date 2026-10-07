"""D: build results/report.md and figures from saved per-topic CSVs (no search is re-run).

Optional inputs, used when present: results/tuned.json (grids, from eval.tune.save_tuned),
results/specificity.json ({qid: specificity}) and results/champion_sweep.json
([{"r":..., "ndcg":..., "median_ms":...}]).
"""
from __future__ import annotations

import json
from pathlib import Path

from eval import plots
from eval.data import select_qids
from eval.runner import read_per_topic_csv
from eval.stats import compare_to_baseline


def _load_variants(results_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    return {p.stem.removeprefix("per_topic_"): read_per_topic_csv(p)
            for p in sorted(results_dir.glob("per_topic_*.csv"))}


def _load_latency(results_dir: Path) -> dict[str, dict[str, float]]:
    out = {}
    for p in sorted(results_dir.glob("summary_*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        out[s["variant"]] = s["latency"]
    return out


def _json_or_none(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build_report(results_dir: str | Path, split: dict, all_qids: list[str], baseline: str = "V0") -> Path:
    results_dir = Path(results_dir)
    figs = results_dir / "figs"
    data = _load_variants(results_dir)
    if not data:
        raise FileNotFoundError(f"no per_topic_*.csv in {results_dir}; run scripts/run_eval.py first")
    sets = {"report half": select_qids("report", split, all_qids), "all 50": select_qids("all", split, all_qids)}
    md = ["# PRISM results", "",
          "Relevant means qrels score >= 1; unjudged documents count as non-relevant, which can understate "
          "every system. Recall@100 is capped well below 1 because every topic has more than 100 relevant documents.",
          "", "## Main table", "", plots.results_table(data, sets), ""]

    judged = {v: {s: plots.mean_over(data[v], q, "Judged@10") for s, q in sets.items()} for v in data}
    md += ["## Judged@10 and condensed nDCG@10 (pooling-bias check)", "",
           "| Variant | Judged@10 (report) | Judged@10 (all 50) | condensed nDCG@10 (report) |", "| --- | --- | --- | --- |"]
    for v in plots._ordered(data):
        md.append(f"| {v} | {judged[v]['report half']:.3f} | {judged[v]['all 50']:.3f} | "
                  f"{plots.mean_over(data[v], sets['report half'], 'cNDCG@10'):.3f} |")
    md.append("")

    if baseline in data and len(data) > 1:
        for label, qids in sets.items():
            rows = compare_to_baseline(data, baseline, "nDCG@10", qids)
            md += [f"## nDCG@10 against {baseline}, {label}", "", plots.stats_table(rows, "nDCG@10", baseline), ""]
        plots.plot_ndcg_bars(data, sets["report half"], figs / "ndcg_bars_report.png",
                             title=f"nDCG@10 by variant, reporting half ({len(sets['report half'])} topics)")
        plots.plot_ndcg_bars(data, sets["all 50"], figs / "ndcg_bars_all.png")
        md += ["![nDCG@10 by variant](figs/ndcg_bars_report.png)", ""]
        if "V2" in data:
            plots.plot_topic_diffs(data, "V2", baseline, sets["all 50"], figs / "diff_V2_V0.png")
            md += [f"![V2 minus {baseline} per topic](figs/diff_V2_V0.png)", ""]

    latency = _load_latency(results_dir)
    if latency:
        md += ["## Latency", "", plots.latency_table(latency), ""]

    spec = _json_or_none(results_dir / "specificity.json")
    if spec and "V2" in data and "V1" in data:
        gains = {q: data["V2"][q]["nDCG@10"] - data["V1"][q]["nDCG@10"] for q in data["V2"] if q in data["V1"]}
        plots.plot_specificity_scatter(spec, gains, figs / "specificity_vs_gain.png")
        md += ["## Query specificity against the gain from authority", "", "![](figs/specificity_vs_gain.png)", ""]

    tuned = _json_or_none(results_dir / "tuned.json")
    if tuned:
        md += ["## Tuned parameters (tuning half only)", ""]
        for name, t in tuned.items():
            md.append(f"- {name}: {t['best_params']}  ({t['metric']} {t['best_mean'][t['metric']]:.4f} on the tuning half)")
            params = list(t["best_params"])
            if len(params) >= 1:
                plots.plot_sensitivity(t["grid"], params[0], figs / f"sens_{name}_{params[0]}.png", t["metric"])
            if len(params) >= 2:
                plots.plot_heatmap(t["grid"], params[0], params[1], figs / f"heat_{name}.png", t["metric"])
        md.append("")

    sweep = _json_or_none(results_dir / "champion_sweep.json")
    if sweep:
        plots.plot_champion_tradeoff(sweep, figs / "champion_tradeoff.png")
        md += ["## Champion list size", "", "![](figs/champion_tradeoff.png)", ""]

    out = results_dir / "report.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    return out
