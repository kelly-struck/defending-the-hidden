# -*- coding: utf-8 -*-
"""Phase 1 (E6) B1/B2 绘图：DSR–FPR 曲线 + τ=0.85 operating point。

读 curve json（默认 results/phase1/matched_fpr_curve.json），
输出 paper/graph/fig_matched_fpr_curve.pdf。
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_LABELS = {
    "full_system": "Full system (multi-agent debate)",
    "single_agent_control": "Single-agent control (asym. rule)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default=r"d:\AgentScope\results\phase1\matched_fpr_curve.json")
    ap.add_argument("--out", default=r"d:\AgentScope\paper\graph\fig_matched_fpr_curve.pdf")
    a = ap.parse_args()

    curve = json.load(open(a.curve, encoding="utf-8"))

    # 按 arm 分组
    arms = {}
    for k, v in curve.items():
        arm = v["arm"]
        arms.setdefault(arm, []).append(v)
    for arm in arms:
        arms[arm].sort(key=lambda x: x["tau"] if x["tau"] is not None else -1)

    plt.figure(figsize=(7, 5))
    for arm, pts in arms.items():
        label = ARM_LABELS.get(arm, arm)
        xs = [p["fpr"] for p in pts]
        ys = [p["dsr"] for p in pts]
        if arm == "full_system":
            plt.plot(xs, ys, marker="o", label=label)
        else:
            plt.scatter(xs, ys, marker="s", s=90, zorder=5, label=label)
        # 标注 τ=0.85 operating point
        op = next((p for p in pts if p.get("tau") is not None and abs(p["tau"] - 0.85) < 1e-6), None)
        if op:
            plt.scatter([op["fpr"]], [op["dsr"]], s=160, facecolors="none", edgecolors="red", zorder=6)
            plt.annotate("τ=0.85\n(operating point)", (op["fpr"], op["dsr"]),
                         textcoords="offset points", xytext=(10, 10), fontsize=9)

    plt.xlabel("False Positive Rate (%) on benign subset (n=200)")
    plt.ylabel("Defense Success Rate (%) on SIUO (n=167)")
    plt.title("DSR–FPR trade-off (matched-FPR comparison)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    print(f"saved -> {out}")
    # 同时存 png 便于快速查看
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150)
    print(f"saved -> {str(out).replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
