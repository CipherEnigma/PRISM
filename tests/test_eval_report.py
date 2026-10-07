"""Plots, tables and the report builder on synthetic per-topic results."""
import json

import pytest

from eval import plots
from eval.report import build_report
from eval.runner import write_per_topic_csv

QIDS = [str(i) for i in range(1, 9)]
SPLIT = {"seed": 42, "tune": QIDS[:4], "report": QIDS[4:]}


def _row(ndcg):
    return {"P@10": ndcg, "nDCG@10": ndcg, "Recall@100": ndcg / 2, "CappedRecall@100": ndcg, "Judged@10": 0.9, "Judged@100": 0.8, "cNDCG@10": ndcg}


def _data():
    return {"V0": {q: _row(0.40) for q in QIDS}, "V1": {q: _row(0.45) for q in QIDS},
            "V2": {q: _row(0.50 if int(q) % 2 else 0.35) for q in QIDS}}


def test_results_table_values():
    t = plots.results_table(_data(), {"report half": QIDS[4:], "all 50": QIDS})
    lines = t.splitlines()
    assert lines[2].startswith("| V0 | 0.400 | 0.400 | 0.200 |")
    assert lines[3].startswith("| V1 | 0.450 |")
    assert len(lines) == 5


def test_stats_and_latency_tables():
    rows = {"V2": {"mean_diff": 0.05, "ci_low": -0.02, "ci_high": 0.12, "wins": 5, "losses": 3, "ties": 0,
                   "p": 0.4, "p_holm": 0.8}}
    s = plots.stats_table(rows)
    assert "+0.050 [-0.020, +0.120]" in s and "5 / 3 / 0" in s and "0.800" in s
    lat = plots.latency_table({"V0": {"median_ms": 12.34, "p95_ms": 30.0}})
    assert "| V0 | 12.3 | 30.0 |" in lat


def test_figures_are_written(tmp_path):
    data = _data()
    assert plots.plot_ndcg_bars(data, QIDS, tmp_path / "bars.png").stat().st_size > 1000
    assert plots.plot_topic_diffs(data, "V2", "V0", QIDS, tmp_path / "diff.png").exists()
    spec = {q: i / 10 for i, q in enumerate(QIDS)}
    gains = {q: (0.1 - spec[q]) for q in QIDS}
    assert plots.plot_specificity_scatter(spec, gains, tmp_path / "scatter.png").exists()
    rows = [{"params": {"beta_max": b, "s_hi": s}, "mean": {"nDCG@10": 0.4 + b - s / 10}}
            for b in (0.0, 0.1, 0.2) for s in (0.6, 0.8)]
    assert plots.plot_specificity_hist(spec, tmp_path / "hist.png", 0.2, 0.8).exists()
    assert plots.plot_sensitivity(rows, "beta_max", tmp_path / "sens.png").exists()
    assert plots.plot_heatmap(rows, "beta_max", "s_hi", tmp_path / "heat.png").exists()
    pts = [{"r": 200, "ndcg": 0.4, "median_ms": 5}, {"r": 1000, "ndcg": 0.45, "median_ms": 20}]
    assert plots.plot_champion_tradeoff(pts, tmp_path / "champ.png").exists()


def test_build_report_end_to_end(tmp_path):
    for v, topics in _data().items():
        write_per_topic_csv(tmp_path / f"per_topic_{v}.csv", topics)
        (tmp_path / f"summary_{v}.json").write_text(json.dumps(
            {"variant": v, "latency": {"median_ms": 10.0, "p95_ms": 20.0, "mean_ms": 11.0}}))
    (tmp_path / "specificity.json").write_text(json.dumps({q: int(q) / 10 for q in QIDS}))
    out = build_report(tmp_path, SPLIT, QIDS)
    text = out.read_text()
    assert "## Main table" in text and "against V0, report half" in text and "## Latency" in text
    assert (tmp_path / "figs" / "ndcg_bars_report.png").exists()
    assert (tmp_path / "figs" / "specificity_vs_gain.png").exists()
    assert (tmp_path / "figs" / "specificity_hist.png").exists() and "Query specificity across" in text


def test_build_report_needs_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_report(tmp_path, SPLIT, QIDS)
