# F1 -- ablation ladder figure (v3 final plan).
# All values are taken verbatim from Table 4 of Defending-the-Hidden-CS-R1.tex
# (same oracle, DeepSeek-V4-Flash, N=167).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# (label, DSR %, colour) -- ascending so that the full system sits on top.
rows = [
    ("No defense",           44.7,  "#7F7F7F"),
    ("Symmetric rule",       48.3,  "#C00000"),
    ("w/o Attacker",         67.7,  "#9DC3E6"),
    ("w/o Intent-as-Action", 73.1,  "#9DC3E6"),
    ("w/o Defender",         75.4,  "#9DC3E6"),
    ("w/o L2 retrieval",     84.4,  "#9DC3E6"),
    ("Full system",          92.22, "#1F4E79"),
]

fig, ax = plt.subplots(figsize=(3.45, 2.55))
n = len(rows)
labs = [r[0] for r in rows]
vals = [r[1] for r in rows]
cols = [r[2] for r in rows]

ax.barh(range(n), vals, color=cols, height=0.62, zorder=3)

ax.set_yticks(range(n))
ax.set_yticklabels(labs, fontsize=6.6)
ax.set_xlim(0, 104)
ax.set_ylim(-0.62, n - 0.30)
ax.set_xlabel("Defense Success Rate (%)", fontsize=7)

for i, v in enumerate(vals):
    txt = ("%.2f" % v).rstrip("0").rstrip(".")
    ax.text(v + 1.6, i, txt, va="center", ha="left", fontsize=6.6,
            color="#1A1A1A", zorder=4)

# Reference line at the undefended model's intrinsic refusal rate.
ax.axvline(44.7, color="#7F7F7F", lw=0.7, ls=(0, (1.6, 1.6)), zorder=2)
ax.text(45.8, n - 0.42, "undefended", fontsize=6.0, color="#7F7F7F", va="center")

ax.tick_params(axis="x", labelsize=6.6, length=2.4)
ax.tick_params(axis="y", length=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#DDDDDD", lw=0.5, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout(pad=0.25)
fig.savefig("figures/fig_ablation_ladder.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("figures/fig_ablation_ladder_preview.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
print("saved pdf+png")
