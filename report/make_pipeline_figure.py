"""Draws the PRISM pipeline diagram for the report (report/fig_pipeline.png)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, QUIET, EDGE, BLUE, TINT = "#0b0b0b", "#52514e", "#8a8985", "#2a78d6", "#e8f0fb"

fig, ax = plt.subplots(figsize=(7.2, 4.1))
ax.set_xlim(0, 100)
ax.set_ylim(-1, 58)
ax.axis("off")


def box(x, y, w, h, title, lines=(), accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", linewidth=1.3 if accent else 1.0,
                                edgecolor=BLUE if accent else EDGE, facecolor=TINT if accent else "white"))
    ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="top", fontsize=7.6, fontweight="bold", color=INK)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 5.2 - i * 2.45, line, ha="center", va="top", fontsize=6.3, color=QUIET)


def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.1, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center", va="center", fontsize=6, color=QUIET, style="italic")


ax.text(1, 56.5, "OFFLINE: build once", fontsize=8.2, fontweight="bold", color=BLUE, va="center")
ax.text(1, 29.5, "ONLINE: every query", fontsize=8.2, fontweight="bold", color=BLUE, va="center")

# offline row
box(1, 37, 19, 17, "Data", ["TREC-COVID corpus", "171,332 papers", "CORD-19 metadata", "OpenAlex citations"])
box(25, 37, 17, 17, "Analyzer", ["lowercase, spelling", "rules, stop words,", "Porter stemming"])
box(47, 37, 17, 17, "Indexer", ["title, abstract and", "all zones; positions;", "champion lists"])
box(69, 35, 30, 20, "On-disk index", ["zone postings + positions", "parametric: year, journal", "norms, lengths, champions", "authority arrays g(d)"], accent=True)
arrow(20.4, 45.5, 24.6, 45.5)
arrow(42.4, 45.5, 46.6, 45.5)
arrow(64.4, 45.5, 68.6, 45.5)

# online row
box(1, 7, 13, 18, "Query", ["free text,", '"phrase",', "AND OR NOT,", "NEAR/k,", "year>=2020"])
box(17, 7, 15, 18, "Parser", ["recursive descent", "ranking terms vs", "restrictions (AST)"])
box(35, 7, 17, 18, "Candidates", ["postings intersected", "shortest first;", "phrase, NEAR,", "filters, champions"])
box(55, 7, 20, 18, "Scorer + net score", ["lnc.ltc cosine per zone", "net = sum(w_z cos_z)", "+ beta(q) g(d)", "BM25 as reference"])
box(78, 7, 21, 18, "Top-K + explainer", ["heap, ties by doc id", "per-term, per-zone", "breakdown with g(d),", "beta(q), net score"])
arrow(14.4, 16, 16.6, 16)
arrow(32.4, 16, 34.6, 16)
arrow(52.4, 16, 54.6, 16)
arrow(75.4, 16, 77.6, 16)

# index is read by the online path
arrow(84, 34.6, 84, 25.2, "reads", lx=3.2, ly=0)
arrow(84, 34.6, 66, 25.2)
arrow(84, 34.6, 44, 25.2)

# evaluation harness
ax.add_patch(FancyBboxPatch((1, 0.2), 98, 5.0, boxstyle="round,pad=0.2,rounding_size=1", linewidth=1.0, edgecolor=EDGE, facecolor="#f6f6f4"))
ax.text(50, 2.7, "Evaluation harness: the same index and search() for V0-V7 and BM25 on 50 TREC-COVID topics (P@10, nDCG@10, Recall@100)",
        ha="center", va="center", fontsize=6.0, color=INK)

out = Path(__file__).with_name("fig_pipeline.png")
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
print("wrote", out)
