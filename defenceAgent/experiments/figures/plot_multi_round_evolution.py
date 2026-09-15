"""
Plot: Multi-Round Self-Evolution Experiment
============================================
Reads results/multi_round_evolution_results.json and produces a 
publication-quality figure for Section 4.6 of the paper.

Figure design:
  - X-axis: 6 phases (Wave A raw, Wave A para, Wave B raw, Wave B para,
                       Wave C raw, Wave C para)
  - Grouped bars: L2 Interception Rate (blue) and L3 Call Rate (orange)
  - Dashed line: DSR (dark red) on same axis — shows safety stability
  - Vertical separators between waves A/B/C
  - Annotations highlighting the L2 learning-curve shift
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # non-interactive backend for PDF export
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
INPUT_FILE  = r"d:\AgentScope\results\multi_round_evolution_results.json"
OUTPUT_FILE = r"d:\AgentScope\paper\graph\multi_round_evolution_plot.pdf"
OUTPUT_PNG  = r"d:\AgentScope\results\multi_round_evolution_plot.png"

# ---------------------------------------------------------------------------
# Academic style
# ---------------------------------------------------------------------------

def setup_style():
    plt.rcParams.update({
        "figure.figsize":    (13, 6),
        "axes.facecolor":    "white",
        "figure.facecolor":  "white",
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linestyle":    "--",
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize":    15,
        "axes.labelsize":    13,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "legend.fontsize":   11,
        "figure.dpi":        300,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_multi_round_evolution():
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] Input file not found: {INPUT_FILE}")
        print("  Please run evaluate_multi_round_evolution.py first.")
        return

    setup_style()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    phases: list[dict] = data.get("phase_metrics", [])
    if not phases:
        print("[Error] No phase metrics found in results file.")
        return

    # ── Prepare arrays ────────────────────────────────────────────────
    # Expected order: A-raw, A-para, B-raw, B-para, C-raw, C-para
    phases_sorted = sorted(
        phases,
        key=lambda m: ("ABC".index(m.get("wave", "A")), 0 if m.get("sub") == "raw" else 1)
    )

    x_labels = []
    l2_rates, l3_rates, dsr_rates = [], [], []
    sub_types = []   # "raw" or "para" — used for bar colour/hatch

    for m in phases_sorted:
        wave = m.get("wave", "?")
        sub  = m.get("sub", "raw")
        tick = f"Wave {wave}\n({'New Attacks' if sub == 'raw' else 'Paraphrase Variants'})"
        x_labels.append(tick)
        l2_rates.append(m.get("l2_rate", 0))
        l3_rates.append(m.get("l3_rate", 0))
        dsr_rates.append(m.get("dsr", 0))
        sub_types.append(sub)

    n = len(x_labels)
    x = np.arange(n)
    width = 0.32

    # ── Figure ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))

    color_l2  = "#2171b5"   # blue
    color_l3  = "#fd8d3c"   # orange
    color_dsr = "#de2d26"   # red

    # Hatch: solid bars for "raw" phases, hatched for "para" (post-evolution)
    def hatch_for(s):
        return "" if s == "raw" else "///"

    bars_l2 = ax.bar(
        x - width / 2, l2_rates, width,
        color=[color_l2 if s == "raw" else "#9ecae1" for s in sub_types],
        hatch=[hatch_for(s) for s in sub_types],
        edgecolor="black", linewidth=0.7,
        label="L2 Interception Rate", zorder=3
    )
    bars_l3 = ax.bar(
        x + width / 2, l3_rates, width,
        color=[color_l3 if s == "raw" else "#fdbe85" for s in sub_types],
        hatch=[hatch_for(s) for s in sub_types],
        edgecolor="black", linewidth=0.7,
        label="L3 Call Rate (Debate)", zorder=3
    )

    # DSR line on the same axis
    ax.plot(
        x, dsr_rates,
        color=color_dsr, lw=2.2, linestyle="--",
        marker="D", markersize=7, markerfacecolor="white",
        markeredgecolor=color_dsr, markeredgewidth=1.8,
        label="DSR (Defense Success Rate)", zorder=4
    )
    # DSR value labels
    for xi, y in zip(x, dsr_rates):
        ax.annotate(
            f"{y:.0%}",
            xy=(xi, y), xytext=(0, 9),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9.5, color=color_dsr, fontweight="bold"
        )

    # Bar value labels (L2 and L3)
    def _add_bar_labels(bars, clr):
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.annotate(
                    f"{h:.0%}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8.5, color=clr, fontweight="bold"
                )

    _add_bar_labels(bars_l2, color_l2)
    _add_bar_labels(bars_l3, color_l3)

    # ── Wave separator lines ──────────────────────────────────────────
    # Separators between wave pairs: after index 1 and after index 3
    for sep_x in [1.5, 3.5]:
        ax.axvline(sep_x, color="gray", linestyle=":", linewidth=1.2, zorder=1)

    # Wave band labels (top of axes)
    wave_bands = [
        (0, 1,   "Wave A\n(Explicit Harm)"),
        (2, 3,   "Wave B\n(Social Harm)"),
        (4, 5,   "Wave C\n(Soft/Implicit Harm)"),
    ]
    y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 1.0 else 1.15
    for lo, hi, label in wave_bands:
        mid = (lo + hi) / 2
        ax.text(
            mid, 1.09, label,
            ha="center", va="bottom",
            fontsize=10, fontstyle="italic", color="#444444",
            transform=ax.get_xaxis_transform()
        )
        # Bracket arrow
        ax.annotate(
            "", xy=(lo - 0.45, 1.06), xytext=(hi + 0.45, 1.06),
            xycoords=ax.get_xaxis_transform(),
            textcoords=ax.get_xaxis_transform(),
            arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.1)
        )

    # ── Axis styling ─────────────────────────────────────────────────
    ax.set_ylabel("Rate", fontsize=13)
    ax.set_xlabel("Experiment Phase", fontsize=13, labelpad=8)
    ax.set_title(
        "Multi-Round Self-Evolution: Continuous Adaptation across Attack Waves",
        fontsize=15, pad=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylim(0, 1.22)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )

    # ── Legend ───────────────────────────────────────────────────────
    solid_patch = mpatches.Patch(facecolor="#bbbbbb", edgecolor="black",
                                 linewidth=0.7, label="Solid bars = New attack items")
    hatch_patch = mpatches.Patch(facecolor="#bbbbbb", edgecolor="black",
                                 linewidth=0.7, hatch="///",
                                 label="Hatched bars = Paraphrase variants (post-evolution)")
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(
        handles + [solid_patch, hatch_patch],
        lbls + ["Solid bars = New attack items",
                "Hatched bars = Paraphrase variants (post-evolution)"],
        loc="upper left", ncol=2, framealpha=0.9, fontsize=9.5
    )

    # ── Save ─────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG,  bbox_inches="tight", dpi=200)
    print(f"[Done] PDF saved  → {OUTPUT_FILE}")
    print(f"[Done] PNG saved  → {OUTPUT_PNG}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plot_multi_round_evolution()
