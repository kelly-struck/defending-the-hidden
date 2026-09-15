# -*- coding: utf-8 -*-
"""Fig 5 (revised): matched-FPR ROC with an explicit cross-oracle control arm.

Series
------
  full_system  @ qwen-plus  (cross-check oracle)  : solid blue   [6 pts]
  single_agent @ qwen-plus  (cross-check oracle)  : solid red    [6 pts]
  single_agent @ DeepSeek-V4-Flash (headline)     : dashed red   [6 pts, NEW]

Rationale
---------
The previous Fig. 5 caption claimed "DeepSeek-V4-Flash oracle" while every data
point came from the qwen-plus cross-check run (N-6). Rather than merely relabel
the caption, we add the DeepSeek run of the *control* arm, which is the arm that
carries the architectural claim: if the control's DSR is oracle-independent, the
~28 pp architecture gap cannot be an artefact of the scoring model.

Also prints a provenance / consistency audit to stdout.
"""
import json
from pathlib import Path
from math import comb
from decimal import Decimal, getcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

getcontext().prec = 60

ROOT = Path(r"D:\AgentScope\ESWA_submission")
CURVE_QWEN = ROOT / "results" / "matched_fpr_curve.json"
CURVE_DS = ROOT / "results" / "matched_fpr_curve_deepseek.json"
OUT_PDF = ROOT / "figs" / "matched_fpr.pdf"
OUT_PNG = ROOT / "figs" / "matched_fpr.png"

N_ATTACK = 167
N_BENIGN = 200

# Headline full-system DSR under the DeepSeek oracle (Table 5, component
# ablation "Full System" row) -- the only DeepSeek full-system number we hold.
DS_FULL_DSR = 92.22
DS_FULL_COUNT = 154


# ----------------------------------------------------------------- helpers
def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return sorted(d.values(), key=lambda v: v["tau"])


def mcnemar_exact_two_sided(b, c):
    """Exact two-sided McNemar: 2 * P(X <= min(b,c)), X ~ Bin(b+c, 0.5)."""
    n, k = b + c, min(b, c)
    s = sum(Decimal(comb(n, i)) for i in range(k + 1))
    return min(float(2 * s / (Decimal(2) ** n)), 1.0)


def mcnemar_bounds(diff, N):
    """Worst / best-case exact p over all admissible discordant splits."""
    worst = None
    for s in range(diff, N + 1):
        if (s - diff) % 2:
            continue
        b, c = (s + diff) // 2, (s - diff) // 2
        p = mcnemar_exact_two_sided(b, c)
        if worst is None or p > worst[2]:
            worst = (b, c, p, s)
    best = (diff, 0, mcnemar_exact_two_sided(diff, 0), diff)
    return worst, best


def count_from_pct(pct, n):
    """Nearest integer count implied by a percentage on n samples."""
    return round(pct / 100.0 * n)


# ----------------------------------------------------------------- audit
qwen_full = [v for v in load(CURVE_QWEN) if v["arm"] == "full_system"]
qwen_single = [v for v in load(CURVE_QWEN) if v["arm"] == "single_agent_control"]
ds_single = load(CURVE_DS)

print("=" * 72)
print("AUDIT 1 -- integer-count reachability (n=167 attack / n=200 benign)")
print("=" * 72)
print(f"{'series':<28}{'tau':>6}{'DSR%':>9}{'k=round':>9}{'k/167':>9}"
      f"{'FPR%':>8}{'k=round':>9}{'k/200':>9}")
for name, series in (("qwen full", qwen_full),
                     ("qwen single", qwen_single),
                     ("deepseek single", ds_single)):
    for v in series:
        kd = count_from_pct(v["dsr"], N_ATTACK)
        kf = count_from_pct(v["fpr"], N_BENIGN)
        print(f"{name:<28}{v['tau']:>6.2f}{v['dsr']:>9.2f}{kd:>9d}"
              f"{kd / N_ATTACK * 100:>9.2f}{v['fpr']:>8.2f}{kf:>9d}"
              f"{kf / N_BENIGN * 100:>9.2f}")

print()
print("=" * 72)
print("AUDIT 2 -- cross-oracle architecture gap at tau = 0.85")
print("=" * 72)
op_q_full = next(v for v in qwen_full if abs(v["tau"] - 0.85) < 1e-6)
op_q_single = next(v for v in qwen_single if abs(v["tau"] - 0.85) < 1e-6)
op_ds_single = next(v for v in ds_single if abs(v["tau"] - 0.85) < 1e-6)

rows = [
    ("DeepSeek-V4-Flash", DS_FULL_COUNT, count_from_pct(op_ds_single["dsr"], N_ATTACK)),
    ("qwen-plus", count_from_pct(op_q_full["dsr"], N_ATTACK), count_from_pct(op_q_single["dsr"], N_ATTACK)),
]
print(f"{'oracle':<22}{'full':>18}{'control':>18}{'delta pp':>11}{'samples':>10}")
gaps = {}
for name, kf, kc in rows:
    f, c = kf / N_ATTACK * 100, kc / N_ATTACK * 100
    gaps[name] = (f, c, f - c, kf - kc)
    print(f"{name:<22}{f:>10.2f}% ({kf:>3d}/167){c:>10.2f}% ({kc:>3d}/167)"
          f"{f - c:>11.2f}{kf - kc:>10d}")

print()
print("=" * 72)
print("AUDIT 3 -- exact McNemar on the architecture gap (two-sided)")
print("=" * 72)
for name, (f, c, gap, nsamp) in gaps.items():
    worst, best = mcnemar_bounds(nsamp, N_ATTACK)
    print(f"{name:<22} diff={nsamp:>3d}  "
          f"worst split b={worst[0]},c={worst[1]} (b+c={worst[3]}) -> p={worst[2]:.3e}   "
          f"best split c=0 -> p={best[2]:.3e}")
    print(f"{'':<22} Holm-adjusted alpha=0.01 -> "
          f"{'SURVIVES' if worst[2] < 0.01 else 'FAILS'} even in the worst case")

print()
print("=" * 72)
print("AUDIT 4 -- control-arm curve agreement across the tau sweep")
print("=" * 72)
devs = []
for a, b in zip(qwen_single, ds_single):
    d = b["dsr"] - a["dsr"]
    devs.append(abs(d))
    print(f"tau={a['tau']:.2f}  qwen {a['dsr']:6.2f}  deepseek {b['dsr']:6.2f}  "
          f"delta {d:+6.2f} pp  ({count_from_pct(b['dsr'], 167) - count_from_pct(a['dsr'], 167):+d} samples)")
print(f"max |delta| = {max(devs):.2f} pp   mean |delta| = {sum(devs) / len(devs):.2f} pp")
print(f"at tau=0.85: {abs(op_ds_single['dsr'] - op_q_single['dsr']):.2f} pp "
      f"(= {abs(count_from_pct(op_ds_single['dsr'], 167) - count_from_pct(op_q_single['dsr'], 167))} sample)")

print()
print("=" * 72)
print("AUDIT 5 -- architecture-gap range across the whole tau sweep (qwen-plus)")
print("=" * 72)
g = [a["dsr"] - b["dsr"] for a, b in zip(qwen_full, qwen_single)]
print(f"gap per tau: " + "  ".join(f"{x:.2f}" for x in g))
print(f"min {min(g):.2f} pp  max {max(g):.2f} pp")

print()
print("=" * 72)
print("AUDIT 6 -- DeepSeek control vs. qwen-plus FULL system (headline sanity)")
print("=" * 72)
print(f"DeepSeek control at tau=0.85 = {op_ds_single['dsr']:.2f}% "
      f"vs qwen-plus full at tau=0.85 = {op_q_full['dsr']:.2f}%")


# ----------------------------------------------------------------- plot
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

C_FULL, C_SINGLE_Q, C_SINGLE_DS = "#1f4e79", "#d62728", "#7f3f98"

fig, ax = plt.subplots(figsize=(7.2, 5.2))

ax.plot([p["fpr"] for p in qwen_full], [p["dsr"] for p in qwen_full],
        "-o", color=C_FULL, linewidth=2.0, markersize=7,
        label="Full system (qwen-plus oracle)")
ax.plot([p["fpr"] for p in qwen_single], [p["dsr"] for p in qwen_single],
        "-s", color=C_SINGLE_Q, linewidth=2.0, markersize=7,
        label="Single-agent control (qwen-plus oracle)")
ax.plot([p["fpr"] for p in ds_single], [p["dsr"] for p in ds_single],
        "--D", color=C_SINGLE_DS, linewidth=1.6, markersize=6,
        markerfacecolor="white", markeredgewidth=1.4,
        label="Single-agent control (DeepSeek-V4-Flash oracle)")

# mark the production operating point on each series; stagger labels so they
# do not collide.
mark_specs = (
    (qwen_full, C_FULL, -50, 9),
    (qwen_single, C_SINGLE_Q, 12, 16),
    (ds_single, C_SINGLE_DS, 14, -22),
)
for pts, col, dx, dy in mark_specs:
    op = next(p for p in pts if abs(p["tau"] - 0.85) < 1e-6)
    ax.scatter([op["fpr"]], [op["dsr"]], s=190, facecolors="none",
               edgecolors="black", linewidths=1.4, zorder=6)
    ax.annotate(r"$\tau$=0.85", (op["fpr"], op["dsr"]),
                textcoords="offset points", xytext=(dx, dy),
                fontsize=8.5, color=col)

ax.set_xlabel("False Positive Rate (%) on matched benign subset")
ax.set_ylabel("Defense Success Rate (%) on SIUO (n=167)")
ax.set_title("Matched-FPR ROC: full system vs. single-agent control")
ax.set_xlim(2.6, 5.0)
ax.set_ylim(60, 95.5)
# the two data clusters are at y in [61.7, 65.9] (controls) and y in
# [90.4, 93.4] (full system), so y in (66, 90) is empty; place the
# legend in the upper-right of that empty band.
ax.legend(loc="upper right", bbox_to_anchor=(0.985, 0.62),
          frameon=True, fancybox=False, edgecolor="black",
          facecolor="white", framealpha=0.95,
          fontsize=8.8, handlelength=2.2)

fig.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.97)
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print()
print(f"saved -> {OUT_PDF}")
print(f"saved -> {OUT_PNG}")
