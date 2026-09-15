"""
Plot: Poisoning Robustness Experiment
======================================
Reads results/poisoning_robustness_results.json and produces a
publication-quality figure + a LaTeX table snippet for Section 4.6.

Figure layout (1×2 subplots):
  Left  – Grouped bar: FPR across three stages (with % annotations)
  Right – Grouped bar: TPR across three stages (attack detection maintained)

The figure tells the story:
  "Poisoning spikes FPR from ~0% to X%, but TTL aging returns it to ~0%
   while TPR remains stable throughout."
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
INPUT_FILE  = r"d:\AgentScope\results\poisoning_robustness_results.json"
OUTPUT_PDF  = r"d:\AgentScope\paper\graph\poisoning_robustness_plot.pdf"
OUTPUT_PNG  = r"d:\AgentScope\results\poisoning_robustness_plot.png"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def setup_style():
    plt.rcParams.update({
        "figure.figsize":    (12, 5.5),
        "axes.facecolor":    "white",
        "figure.facecolor":  "white",
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linestyle":    "--",
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "xtick.labelsize":   10.5,
        "ytick.labelsize":   10.5,
        "legend.fontsize":   10,
        "figure.dpi":        300,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bar_label(ax, bar, fmt=".1%"):
    h = bar.get_height()
    ax.annotate(
        format(h, fmt),
        xy=(bar.get_x() + bar.get_width() / 2, h),
        xytext=(0, 4), textcoords="offset points",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold",
    )

def _make_bar_chart(ax, x_labels, values, colors, title, ylabel,
                    ylim=(0, 1.18), highlight_idx=1, highlight_color="#de2d26"):
    """Draw a single grouped-bar chart on `ax`."""
    x = np.arange(len(x_labels))
    bars = []
    for i, (val, clr) in enumerate(zip(values, colors)):
        edge = highlight_color if i == highlight_idx else "black"
        lw   = 2.0           if i == highlight_idx else 0.8
        bar = ax.bar(x[i], val, width=0.5, color=clr,
                     edgecolor=edge, linewidth=lw, zorder=3)
        bars.append(bar)
        _bar_label(ax, bar[0])

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    return bars

# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------
def plot_poisoning_robustness():
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] {INPUT_FILE} not found. Run evaluate_poisoning_robustness.py first.")
        return

    setup_style()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    table = data["table"]
    cfg   = data["config"]

    # Parse table rows (order: clean, poisoned, post-TTL)
    stages   = [r["stage"].replace("(", "\n(") for r in table]
    fpr_vals = [r["fpr"] for r in table]
    tpr_vals = [r.get("tpr") or 0.0 for r in table]
    sizes    = [r["collection_size"] for r in table]

    # Use short labels for X axis
    x_labels_short = [
        "Clean Base",
        "After\nPoisoning",
        "After\nTTL Aging",
    ]

    # Colour scheme
    # FPR: green (low) → red (high) → green (low again)
    fpr_colors = ["#74c476", "#de2d26", "#74c476"]
    # TPR: stable blue
    tpr_colors = ["#9ecae1", "#4292c6", "#2171b5"]

    fig, (ax_fpr, ax_tpr) = plt.subplots(1, 2, figsize=(12, 5.5))

    # ── Left: FPR ─────────────────────────────────────────────────────
    _make_bar_chart(
        ax_fpr, x_labels_short, fpr_vals, fpr_colors,
        title="False Positive Rate (FPR)\nBenign Queries Incorrectly Blocked",
        ylabel="FPR",
        highlight_idx=1,
    )

    # Annotation arrow: "TTL reduces FPR"
    ax_fpr.annotate(
        "TTL Aging\nreduces FPR",
        xy=(2, fpr_vals[2] + 0.04),
        xytext=(1.5, fpr_vals[2] + 0.25),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
        fontsize=9, color="#555555", ha="center",
    )

    # ── Right: TPR ────────────────────────────────────────────────────
    _make_bar_chart(
        ax_tpr, x_labels_short, tpr_vals, tpr_colors,
        title="True Positive Rate (TPR)\nAttack Detection Maintained",
        ylabel="TPR",
        highlight_idx=2,
        highlight_color="#2171b5",
    )

    # Annotation: "TPR stable"
    ax_tpr.annotate(
        "TPR remains\nstable",
        xy=(2, tpr_vals[2] + 0.04),
        xytext=(1.5, min(tpr_vals) - 0.12),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
        fontsize=9, color="#555555", ha="center",
    )

    # ── Shared annotations ────────────────────────────────────────────
    fig.suptitle(
        f"Poisoning Robustness: TTL/Aging Mechanism Effectiveness\n",
        fontsize=12, y=1.01,
    )

    # ── Collection-size annotation below each bar ─────────────────────
    for ax in (ax_fpr, ax_tpr):
        for xi, sz in enumerate(sizes):
            ax.text(
                xi, -0.10, f"n={sz}",
                ha="center", va="top", fontsize=8.5,
                color="#666666",
                transform=ax.get_xaxis_transform(),
            )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    Path(OUTPUT_PDF).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=200)
    print(f"[Done] PDF → {OUTPUT_PDF}")
    print(f"[Done] PNG → {OUTPUT_PNG}")
    plt.close(fig)

    # ── Print LaTeX table snippet ────────────────────────────────────
    _print_latex_table(table, cfg)


def _print_latex_table(table: list[dict], cfg: dict):
    """Print a ready-to-paste LaTeX table for the paper."""
    print("\n" + "─" * 55)
    print("LaTeX Table Snippet (paste into paper):")
    print("─" * 55)
    print(r"""\begin{table}[h]
\centering
\caption{Poisoning Robustness of the Vector Library under """ +
          f"{cfg['poison_ratio']} benign-label injection." +
          r"""}
\label{tab:poisoning_robustness}
\begin{tabular}{lccl}
\toprule
\textbf{Stage} & \textbf{Lib. Size} & \textbf{FPR↓} & \textbf{TPR↑} \\
\midrule""")
    for row in table:
        stage = row["stage"].replace("(", "(").replace("→", r"$\to$")
        sz    = row["collection_size"]
        fpr   = f"{row['fpr']:.1%}"
        tpr   = f"{row['tpr']:.1%}" if row["tpr"] is not None else "—"
        print(fr"{stage} & {sz} & {fpr} & {tpr} \\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plot_poisoning_robustness()
