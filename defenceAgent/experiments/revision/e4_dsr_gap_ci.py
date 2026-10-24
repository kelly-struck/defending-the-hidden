# -*- coding: utf-8 -*-
"""EX-E4-1 — Confidence interval for the 3.6 pp top-of-ranking DSR gap.

Reviewer R1-3 / AE-(iii): the headline "+3.6 pp over the strongest baseline"
(154/167 vs 148/167) is presented with paradigm-level language; we quantify it.

Method (per Roadmap §七 7.0):
  * Independent difference of proportions  -> Newcombe (1998) method 10,
    built from two Wilson score intervals.  This is the canonical exact-ish
    interval for p1 - p2 with independent samples.
  * Paired view (same 167 samples, same oracle) -> we do NOT observe the
    discordant counts (b, c); we report the worst-case and best-case exact
    two-sided McNemar p over all admissible splits with |b - c| = 6,
    mirroring the paper's existing McNemar framing (L1053).

No API calls. Pure arithmetic on the published counts.

Output: dsr_gap_ci.json (workspace root, per guide) + console verdict.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scipy import stats

Z = 1.959963984540054  # 95%

# ---- published counts (paper L470 / Table 1) ----
N = 167
A_HIT, A_N = 154, 167          # our framework, primary victim model
B_HIT, B_N = 148, 167          # strongest baseline = Safety Prompt (88.6%)
L3_TPR, L3_FPR = 154 / 167, 36 / 1000   # DSR 92.22%, FPR 3.60%


def wilson(k: int, n: int, z: float = Z):
    """Wilson score interval (proportion)."""
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, centre - half, centre + half


def newcombe_diff(k1: int, n1: int, k2: int, n2: int, z: float = Z):
    """Newcombe (1998) method 10 interval for p1 - p2 (independent samples)."""
    p1, l1, u1 = wilson(k1, n1, z)
    p2, l2, u2 = wilson(k2, n2, z)
    diff = p1 - p2
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return diff, lo, hi, (l1, u1), (l2, u2)


def mcnemar_bounds(discordant: int):
    """Worst/best exact two-sided McNemar p over admissible splits |b-c|=d."""
    if discordant == 0:
        return {"worst_case_p": 1.0, "best_case_p": 1.0, "worst_split": [0, 0],
                "best_split": [0, 0]}
    b, c = discordant, 0            # most favourable: all discordant favour us
    best = float(stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
    # worst case: minimises two-sided binomial p over |b-c| fixed
    split_worst, p_worst = None, 1.0
    for bb in range(discordant, 2 * discordant + 1):
        cc = bb - discordant
        p = float(stats.binomtest(bb, bb + cc, 0.5, alternative="two-sided").pvalue)
        if p < p_worst:
            p_worst, split_worst = p, [bb, cc]
    return {"worst_case_p": float(p_worst), "best_case_p": float(best),
            "worst_split": split_worst, "best_split": [b, c],
            "discordant_total": discordant}


def main() -> None:
    # ---------- E4b: independent Newcombe interval on the 3.6 pp gap ----------
    diff, lo, hi, ci_a, ci_b = newcombe_diff(A_HIT, A_N, B_HIT, B_N)
    includes_zero = (lo <= 0.0 <= hi)
    # equivalence-style: does the interval also sit entirely below some margin?
    m = mcnemar_bounds(6)   # 154 vs 148 -> gap = 6 samples

    pa, la, ua = wilson(A_HIT, A_N)
    pb, lb, ub = wilson(B_HIT, B_N)

    report = {
        "protocol": "EX-E4-1 DSR top-of-ranking gap CI (Neurocomputing NEUCOM-D-26-20854)",
        "comparison": "full framework (154/167) vs strongest baseline Safety Prompt (148/167)",
        "n": N,
        "counts": {"framework": [A_HIT, A_N], "baseline": [B_HIT, B_N]},
        "independent_difference": {
            "method": "Newcombe (1998) method 10, from two Wilson score intervals",
            "point_estimate_pp": round(diff * 100, 4),
            "ci95_pp": [round(lo * 100, 4), round(hi * 100, 4)],
            "ci95_prop": [lo, hi],
            "includes_zero": includes_zero,
            "framework_wilson_ci95_pp": [round(la * 100, 4), round(ua * 100, 4)],
            "baseline_wilson_ci95_pp": [round(lb * 100, 4), round(ub * 100, 4)],
            "half_width_pp": round((hi - lo) / 2 * 100, 4),
        },
        "paired_view": {
            "note": "same 167 samples + same oracle, but discordant (b,c) unobserved; "
                    "exact McNemar bounded over all admissible splits with |b-c|=6.",
            **m,
            "worst_case_p_ge_holm_threshold": m["worst_case_p"] >= 0.01,
        },
        "power_note": {
            "n": N,
            "gap_samples": 6,
            "comment": "At N=167 a 6/167 gap is underpowered for a 3.6 pp effect; "
                       "the experimental unit is the sample, so the interval is wide.",
        },
        "statistical_verdict": (
            f"95% CI for the gap is [{lo*100:.2f}, {hi*100:.2f}] pp and "
            f"{'CONTAINS' if includes_zero else 'EXCLUDES'} 0 "
            f"(point estimate {diff*100:.2f} pp). "
            "The top-of-ranking margin is therefore NOT statistically established at "
            "N=167; report the ranking as DESCRIPTIVE and keep the statistical claim on "
            "the controlled debate-vs-single-agent comparison (TX-C3)."
            if includes_zero else
            f"95% CI for the gap is [{lo*100:.2f}, {hi*100:.2f}] pp and EXCLUDES 0 "
            f"(point estimate {diff*100:.2f} pp). The margin is statistically "
            "distinguishable from zero at N=167."),
        "pipeline_consistency": {
            "note": "cross-check: DSR 154/167=92.22%, FPR 36/1000=3.60% match Table 7 header",
            "dsr": round(L3_TPR, 6), "fpr": round(L3_FPR, 6),
        },
    }
    out = Path(__file__).resolve().parent / "dsr_gap_ci.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "point_pp": report["independent_difference"]["point_estimate_pp"],
        "ci95_pp": report["independent_difference"]["ci95_pp"],
        "includes_zero": includes_zero,
        "worst_case_mcnemar_p": round(m["worst_case_p"], 6),
        "best_case_mcnemar_p": float(f"{m['best_case_p']:.3e}"),
    }, ensure_ascii=False, indent=1))
    print("\nVERDICT:", report["statistical_verdict"])
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
