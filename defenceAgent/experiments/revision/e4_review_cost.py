# -*- coding: utf-8 -*-
"""EX-E4-2 — Review-cost quantification for the 99%-benign deployment mix.

Reviewer R2-3 / AE-(iii): at r=0.99 the PPV collapses to 20.56% and FP:TP=3.86:1,
implying alert fatigue. This script converts the measured FPR/DSR into operational
numbers per **1,000 queries**: alerts (TP+FP), true hits (TP), benign false alarms
(FP), and human review time under a stated 3 min/alert assumption.

Deployment prevalence axis r from Table 7 (benign share):
    r = 0.50, 0.90, 0.99, 0.999

Definitions (per 1,000 queries, of which r*1000 benign and (1-r)*1000 attacks):
    TP = attacks * DSR                      (genuine attacks intercepted)
    FP = benign  * FPR                      (benign queries flagged)
    alerts = TP + FP
    review_hours = alerts * 3 min / 60

NOTE: Table 7's PPV/FP:TP are conditional on *matched* (routed-to-L2) queries,
whereas the 3.60% FPR is the benign-conditional rate on the full 1,000-query
benign benchmark. Both are reported explicitly so the paper can state which
quantity each number refers to. We report the operational view.

No API calls. Output: review_cost_table.md (+ review_cost_table.csv).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

DSR = 154 / 167          # 92.22%
FPR = 36 / 1000          # 3.60%
MIN_PER_ALERT = 3.0      # stated assumption
R_GRID = [0.50, 0.90, 0.99, 0.999]
N_QUERIES = 1000

# Table 7 published PPV / FP:TP (matched-query view, for cross-check only)
TABLE7 = {
    0.500: {"ppv": 96.24, "fp_tp": 0.04},
    0.900: {"ppv": 74.00, "fp_tp": 0.35},
    0.990: {"ppv": 20.56, "fp_tp": 3.86},
    0.999: {"ppv": 2.50, "fp_tp": 39.00},
}


def row_for(r: float) -> dict:
    benign = r * N_QUERIES
    attacks = (1 - r) * N_QUERIES
    tp = attacks * DSR
    fp = benign * FPR
    alerts = tp + fp
    ppv = tp / alerts if alerts > 0 else float("nan")
    fp_tp = fp / tp if tp > 0 else float("inf")
    review_min = alerts * MIN_PER_ALERT
    return {
        "r": r,
        "queries": N_QUERIES,
        "benign": benign,
        "attacks": attacks,
        "TP": tp,
        "FP": fp,
        "alerts_TP_plus_FP": alerts,
        "PPV_pct": ppv * 100,
        "FP_per_TP": fp_tp,
        "review_min": review_min,
        "review_hours": review_min / 60.0,
    }


def main() -> None:
    rows = [row_for(r) for r in R_GRID]

    # self-consistency cross-check against Table 7 (matched-query view)
    checks = []
    for r in R_GRID:
        row = next(x for x in rows if abs(x["r"] - r) < 1e-12)
        t7 = TABLE7[r]
        checks.append({
            "r": r,
            "ppv_operational_pct": round(row["PPV_pct"], 2),
            "ppv_table7_matched_pct": t7["ppv"],
            "ppv_delta_pp": round(row["PPV_pct"] - t7["ppv"], 2),
            "fp_tp_operational": round(row["FP_per_TP"], 2),
            "fp_tp_table7": t7["fp_tp"],
            "note": "operational PPV from DSR/FPR reproduces Table 7 exactly (Δ=0.00 pp); "
                    "Table 7's PPV is conditional on matched (routed) queries but coincides "
                    "with the all-benign view at the reported precision.",
        })

    # ---------- markdown ----------
    md = [
        "# EX-E4-2 — Review cost per 1,000 queries (deployment prevalence)",
        "",
        f"- Operating point: DSR = {DSR*100:.2f}% (154/167), FPR = {FPR*100:.2f}% (36/1,000 benign)",
        f"- Review-time assumption: **{MIN_PER_ALERT:.0f} min per alert** (stated in text)",
        "- Per 1,000 queries: `attacks = 1000*(1-r)`, `benign = 1000*r`",
        "",
        "| benign share r | attacks | alerts (TP+FP) | TP (true hits) | FP (false alarms) | PPV | review time |",
        "|---|---|---|---|---|---|---|",
    ]
    for x in rows:
        md.append("| {:.3f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.2f}% | {:.0f} min ({:.1f} h) |".format(
            x["r"], x["attacks"], x["alerts_TP_plus_FP"], x["TP"], x["FP"],
            x["PPV_pct"], x["review_min"], x["review_hours"]))
    md += [
        "",
        "## Notes for the manuscript (Table 7 / §5.3.2 / §5.7)",
        "",
        "1. **Base-rate phenomenon, not architecture defect.** At r=0.99 a detector with "
        "FPR 3.6% facing 1% attack prevalence necessarily produces FP:TP ≈ 3.9:1; "
        "this holds for any detector with the same FPR/DSR.",
        "2. **Operational cost.** At r=0.99, {:.0f} alerts per 1,000 queries ⇒ ≈ {:.1f} h of "
        "review at {:.0f} min/alert. At r=0.999 the same FPR yields far fewer alerts "
        "({:.0f}/1,000) but a PPV of only {:.2f}%.".format(
            next(x for x in rows if x["r"] == 0.99)["alerts_TP_plus_FP"],
            next(x for x in rows if x["r"] == 0.99)["review_hours"],
            MIN_PER_ALERT,
            next(x for x in rows if x["r"] == 0.999)["alerts_TP_plus_FP"],
            next(x for x in rows if x["r"] == 0.999)["PPV_pct"]),
        "3. **Mitigations to discuss.** (a) sampled review; (b) L3-confidence-prioritised "
        "queue; (c) precision-favouring τ; (d) tiered triage. Deployment boundary remains "
        "**offline pre-deployment audit**, not autonomous real-time blocking.",
        "4. **Alert volume saturates at the FPR floor.** Review time is *non-monotonic* in "
        "r: it is dominated by the benign-scan cost once benign volume is large, so alerts "
        "peak around 479/1,000 (r=0.5) and fall to 37/1,000 at r=0.999 — while PPV collapses. "
        "The operational problem at high r is therefore **triage precision**, not raw volume.",
        "",
        "## Self-consistency cross-check vs Table 7 (matched-query view)",
        "",
        "| r | PPV operational | PPV Table 7 (matched) | Δ pp | FP:TP operational | FP:TP Table 7 |",
        "|---|---|---|---|---|---|",
    ]
    for c in checks:
        md.append("| {:.3f} | {:.2f}% | {:.2f}% | {:+.2f} | {:.2f} | {:.2f} |".format(
            c["r"], c["ppv_operational_pct"], c["ppv_table7_matched_pct"],
            c["ppv_delta_pp"], c["fp_tp_operational"], c["fp_tp_table7"]))
    md += ["", "> **Self-consistency:** the operational PPV computed directly from DSR/FPR "
              "reproduces Table 7 to the reported precision (Δ = 0.00 pp on all four rows), "
              "confirming that Table 7 follows from the measured operating point. The "
              "review-cost columns are the new operational quantities.", ""]

    root = Path(__file__).resolve().parent
    (root / "review_cost_table.md").write_text("\n".join(md), encoding="utf-8")
    with (root / "review_cost_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for x in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in x.items()})

    summary = {
        "protocol": "EX-E4-2 review cost per 1,000 queries",
        "assumptions": {"min_per_alert": MIN_PER_ALERT, "dsr": DSR, "fpr": FPR,
                        "r_grid": R_GRID, "n_queries": N_QUERIES},
        "rows": rows,
        "table7_crosscheck": checks,
    }
    (root / "review_cost_table.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'r':>6} {'alerts':>8} {'TP':>7} {'FP':>7} {'PPV%':>7} {'hrs':>6}")
    for x in rows:
        print(f"{x['r']:>6.3f} {x['alerts_TP_plus_FP']:>8.1f} {x['TP']:>7.1f} "
              f"{x['FP']:>7.1f} {x['PPV_pct']:>7.2f} {x['review_hours']:>6.2f}")
    print(f"\nsaved -> {root/'review_cost_table.md'}")
    print(f"saved -> {root/'review_cost_table.csv'}")


if __name__ == "__main__":
    main()
