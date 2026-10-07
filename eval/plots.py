"""D: results table, nDCG@10 bars, per-topic differences, specificity vs gain, sensitivity, bootstrap CIs."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from eval.stats import spearman_ci       # noqa: E402

PerTopic = Mapping[str, Mapping[str, Mapping[str, float]]]    # variant -> qid -> metric -> value
METRICS = ["P@10", "nDCG@10", "Recall@100"]
ORDER = ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "R"]
BASELINES = {"V0", "R"}

# Reference categorical slots 1 and 2 (blue, orange) and neutral inks.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9893"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2dd"


def _style(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def _ordered(variants) -> list[str]:
    return [v for v in ORDER if v in variants] + sorted(v for v in variants if v not in ORDER)


def mean_over(per_topic: Mapping[str, Mapping[str, float]], qids: Sequence[str], metric: str) -> float:
    vals = [per_topic[q][metric] for q in qids if q in per_topic]
    return float(np.mean(vals)) if vals else float("nan")


def results_table(data: PerTopic, qid_sets: Mapping[str, Sequence[str]]) -> str:
    """Markdown table: one row per variant, P@10 / nDCG@10 / Recall@100 for each topic set."""
    head = "| Variant | " + " | ".join(f"{s}: {m}" for s in qid_sets for m in METRICS) + " |"
    rule = "| --- |" + " --- |" * (len(qid_sets) * len(METRICS))
    lines = [head, rule]
    for v in _ordered(data):
        cells = [f"{mean_over(data[v], qids, m):.3f}" for qids in qid_sets.values() for m in METRICS]
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def stats_table(rows: Mapping[str, Mapping], metric: str = "nDCG@10", baseline: str = "V0") -> str:
    """Markdown table of mean difference vs the baseline with CI, wins/losses/ties and Holm p."""
    lines = [f"| Variant | {metric} diff vs {baseline} (95% CI) | W / L / T | sign test p | Holm p |",
             "| --- | --- | --- | --- | --- |"]
    for v in _ordered(rows):
        r = rows[v]
        lines.append(f"| {v} | {r['mean_diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                     f"{r['wins']} / {r['losses']} / {r['ties']} | {r['p']:.3f} | {r['p_holm']:.3f} |")
    return "\n".join(lines)


def latency_table(summaries: Mapping[str, Mapping[str, float]]) -> str:
    """Markdown table of median and p95 latency in ms per variant."""
    lines = ["| Variant | median (ms) | p95 (ms) |", "| --- | --- | --- |"]
    for v in _ordered(summaries):
        s = summaries[v]
        lines.append(f"| {v} | {s['median_ms']:.1f} | {s['p95_ms']:.1f} |")
    return "\n".join(lines)


def plot_ndcg_bars(data: PerTopic, qids: Sequence[str], path: str | Path, metric: str = "nDCG@10",
                   title: str | None = None) -> Path:
    """One bar per variant. Baselines (V0, R) are gray, PRISM variants blue; values on the bars."""
    names = _ordered(data)
    vals = [mean_over(data[v], qids, metric) for v in names]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(names, vals, color=[GRAY if v in BASELINES else BLUE for v in names], width=0.6)
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom",
                fontsize=8, color=INK2)
    _style(ax, ylabel=metric, title=title or f"{metric} by variant ({len(qids)} topics)")
    ax.set_ylim(0, max(vals + [0.01]) * 1.15)
    return _save(fig, path)


def plot_topic_diffs(data: PerTopic, a: str, b: str, qids: Sequence[str], path: str | Path,
                     metric: str = "nDCG@10") -> Path:
    """Sorted per-topic differences a - b: wins (blue) above zero, losses (orange) below."""
    diffs = sorted((data[a][q][metric] - data[b][q][metric] for q in qids if q in data[a] and q in data[b]))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [BLUE if d > 0 else ORANGE if d < 0 else GRAY for d in diffs]
    ax.bar(range(len(diffs)), diffs, color=colors, width=0.7)
    ax.axhline(0, color=INK2, linewidth=0.8)
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    _style(ax, xlabel="topics, sorted by difference", ylabel=f"{metric}: {a} minus {b}",
           title=f"{a} vs {b}: {wins} wins, {losses} losses, mean {np.mean(diffs):+.3f}")
    ax.set_xticks([])
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=[f"{a} better", f"{b} better"], frameon=False, fontsize=8)
    return _save(fig, path)


def plot_specificity_scatter(specificity: Mapping[str, float], gains: Mapping[str, float],
                             path: str | Path, gain_label: str = "nDCG@10 gain from authority (V2 - V1)") -> Path:
    """Query specificity against the per-topic gain of authority: the picture behind the N2 story."""
    qids = sorted(set(specificity) & set(gains))
    x = np.array([specificity[q] for q in qids])
    y = np.array([gains[q] for q in qids])
    rho, lo, hi = spearman_ci(x, y)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=28, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(0, color=INK2, linewidth=0.8)
    if len(x) > 1 and x.std() > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, slope * xs + intercept, color=ORANGE, linewidth=1.5)
    _style(ax, xlabel="query specificity (mean normalized idf)", ylabel=gain_label,
           title=f"Spearman rho = {rho:+.2f} (95% CI {lo:+.2f} to {hi:+.2f}), {len(x)} topics")
    return _save(fig, path)


def plot_specificity_hist(specificity: Mapping[str, float], path: str | Path, s_lo: float | None = None,
                          s_hi: float | None = None) -> Path:
    """Histogram of the query specificity of every topic. The gate can only work if this spreads out;
    s_lo and s_hi (where beta starts to fall and reaches zero) are drawn when given."""
    vals = np.array(list(specificity.values()))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(vals, bins=min(15, max(5, len(vals) // 3)), color=BLUE, edgecolor="white", linewidth=1)
    for x, label in ((s_lo, "s_lo"), (s_hi, "s_hi")):
        if x is not None:
            ax.axvline(x, color=ORANGE, linewidth=1.5)
            ax.text(x, ax.get_ylim()[1] * 0.95, f" {label}", color=INK2, fontsize=8, va="top")
    _style(ax, xlabel="query specificity (mean normalized idf)", ylabel="topics",
           title=f"Specificity of {len(vals)} topics: min {vals.min():.2f}, median {np.median(vals):.2f}, max {vals.max():.2f}")
    return _save(fig, path)


def plot_sensitivity(rows: Sequence[Mapping], param: str, path: str | Path, metric: str = "nDCG@10") -> Path:
    """Tuning-half score against one parameter (other parameters at their best), from a saved grid."""
    best = max(rows, key=lambda r: r["mean"][metric])
    series = sorted((r["params"][param], r["mean"][metric]) for r in rows
                    if all(r["params"][k] == best["params"][k] for k in r["params"] if k != param))
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot([s[0] for s in series], [s[1] for s in series], color=BLUE, linewidth=2, marker="o", markersize=5)
    _style(ax, xlabel=param, ylabel=f"{metric} (tuning half)", title=f"Sensitivity to {param}")
    return _save(fig, path)


def plot_heatmap(rows: Sequence[Mapping], x: str, y: str, path: str | Path, metric: str = "nDCG@10") -> Path:
    """Two-parameter surface (e.g. beta_max by s_hi) in one sequential hue; best cell is outlined."""
    xs = sorted({r["params"][x] for r in rows})
    ys = sorted({r["params"][y] for r in rows})
    grid = np.full((len(ys), len(xs)), np.nan)
    for r in rows:
        grid[ys.index(r["params"][y]), xs.index(r["params"][x])] = r["mean"][metric]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(grid, origin="lower", cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(xs)), [str(v) for v in xs])
    ax.set_yticks(range(len(ys)), [str(v) for v in ys])
    iy, ix = np.unravel_index(np.nanargmax(grid), grid.shape)
    ax.add_patch(plt.Rectangle((ix - 0.5, iy - 0.5), 1, 1, fill=False, edgecolor=ORANGE, linewidth=2))
    for (j, i), v in np.ndenumerate(grid):
        if not np.isnan(v):
            ax.text(i, j, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if v > np.nanmean(grid) + 0.5 * np.nanstd(grid) else INK)
    ax.set_xlabel(x, color=INK2)
    ax.set_ylabel(y, color=INK2)
    ax.set_title(f"{metric} (tuning half)", color=INK, fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, path)


def plot_champion_tradeoff(points: Sequence[Mapping], path: str | Path) -> Path:
    """Champion size r: quality (nDCG@10) against median latency. points: r, ndcg, median_ms."""
    pts = sorted(points, key=lambda p: p["r"])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot([p["median_ms"] for p in pts], [p["ndcg"] for p in pts], color=BLUE, linewidth=2,
            marker="o", markersize=6)
    for p in pts:
        ax.annotate(f"r={p['r']}", (p["median_ms"], p["ndcg"]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color=INK2)
    _style(ax, xlabel="median latency (ms)", ylabel="nDCG@10", title="Champion size: quality vs speed")
    return _save(fig, path)
