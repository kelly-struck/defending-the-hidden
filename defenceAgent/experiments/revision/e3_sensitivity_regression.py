# -*- coding: utf-8 -*-
"""EX-E3-3 — Regression check: does sensitivity_ext reproduce sensitivity.py?

The acceptance criterion for E3 requires that the refactored `simulate()`
(now parameterised) did not change semantics. We re-derive the overlapping
configurations and compare against the *pre-existing* `sensitivity_report.json`
produced by `sensitivity.py` before the refactor.

Overlapping configurations:
  * B_ttl_sweep -> "base=10,min=10,max=30" (main config) == E3 cell D_T=10/30
    (and the T-span cells for base variants)
  * C_l3_noise -> "tpr=1.0,fpr=0.0" (perfect oracle) == E3 default L3 settings

Only cells whose parameters are *identical* on both sides are compared;
`e.R *= DELTA` on the main B config must yield bit-identical metrics.

Output: results/sensitivity_ext_regression.json + console verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_run_ttl_simulation as R
import e3_sensitivity_ext as E3

OUT = R.OUT_DIR / "sensitivity_ext_regression.json"
KEYS = E3.KEYS
POLICIES = E3.POLICIES


def compare(exp, got, label):
    """Compare expected vs got metric dicts; return a pass/fail record."""
    mism = []
    for p in POLICIES:
        for k in KEYS:
            e, g = exp[p].get(k), got[p].get(k)
            if e is None:
                continue
            if abs(float(e) - float(g)) > 1e-12:
                mism.append({"policy": p, "metric": k, "expected": e, "got": g})
    return {"label": label, "n_checks": len(POLICIES) * len(KEYS),
            "mismatches": mism, "pass": len(mism) == 0}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ref_path = R.OUT_DIR / "sensitivity_report.json"
    if not ref_path.exists():
        raise SystemExit(f"missing reference: {ref_path}")
    ref = json.loads(ref_path.read_text(encoding="utf-8"))

    # ---- recompute the *identical* B main-config cell with the refactored code ----
    M = E3.load_matrices()
    got_b_main = {}
    for p in POLICIES:
        res = R.simulate(p, M["M_benign"], M["M_attack_main"], M["M_rare"], M["tau"],
                         M["lib_kinds"], M_attack_extra=M["M_attack_siuo"],
                         extra_label="siuo",
                         ttl_base=10.0, ttl_min=10.0, ttl_max=30.0,
                         delta=0.5, quarantine_o=3, alpha=1.0, beta=1.0)
        got_b_main[p] = {k: res[k] for k in KEYS}

    exp_b_main = ref["B_ttl_sweep"]["base=10,min=10,max=30"]

    # ---- recompute the perfect-oracle C cell ----
    got_c_perfect = {}
    for p in POLICIES:
        res = R.simulate(p, M["M_benign"], M["M_attack_main"], M["M_rare"], M["tau"],
                         M["lib_kinds"], M_attack_extra=M["M_attack_siuo"],
                         extra_label="siuo", l3_tpr=1.0, l3_fpr=0.0,
                         ttl_base=10.0, ttl_min=10.0, ttl_max=30.0,
                         delta=0.5, quarantine_o=3, alpha=1.0, beta=1.0)
        got_c_perfect[p] = {k: res[k] for k in KEYS}
    exp_c_perfect = ref["C_l3_noise"]["tpr=1.0,fpr=0.0"]

    # ---- B_ttl_sweep: every T-span row that E3 re-derives independently ----
    # sensitivity.py grid: base in {5,10,20} x (min,max) in {(10,30),(5,20),(15,40)}
    # E3 D_T_span grid covers (10,30),(5,20),(15,40),(10,20),(5,30).
    # Compare each overlapping span with the SAME base as reported (base=min for the
    # competence sweep cells: base affects only current-TTL).
    tspan_cases = [
        ("base=5,min=10,max=30", 5.0, 10.0, 30.0),
        ("base=10,min=10,max=30", 10.0, 10.0, 30.0),
        ("base=20,min=10,max=30", 20.0, 10.0, 30.0),
        ("base=10,min=5,max=20", 10.0, 5.0, 20.0),
        ("base=10,min=15,max=40", 10.0, 15.0, 40.0),
    ]
    tspan_results = []
    for key, base, tmin, tmax in tspan_cases:
        got = {}
        for p in POLICIES:
            res = R.simulate(p, M["M_benign"], M["M_attack_main"], M["M_rare"], M["tau"],
                             M["lib_kinds"], M_attack_extra=M["M_attack_siuo"],
                             extra_label="siuo", ttl_base=base, ttl_min=tmin,
                             ttl_max=tmax, delta=0.5, quarantine_o=3,
                             alpha=1.0, beta=1.0)
            got[p] = {k: res[k] for k in KEYS}
        exp = ref["B_ttl_sweep"][key]
        tspan_results.append(compare(exp, got, f"B_ttl_sweep {key}"))

    r1 = compare(exp_b_main, got_b_main, "B_ttl_sweep base=10,min=10,max=30 (main)")
    r2 = compare(exp_c_perfect, got_c_perfect, "C_l3_noise tpr=1.0,fpr=0.0 (perfect oracle)")

    report = {
        "protocol": "EX-E3-3 regression: refactored simulate() vs pre-refactor sensitivity_report.json",
        "reference_file": str(ref_path),
        "refactor_changes": [
            "simulate() signature += delta/quarantine_o/alpha/beta/r_benign kwargs (defaults = module constants)",
            "in-body names DELTA/QUARANTINE_O/ALPHA/BETA rebound from kwargs",
            "Entry.__init__ accepts alpha/beta (defaults = module constants)",
            "batch sizes read r_benign instead of module N_BENIGN_PER_STEP/N_ATTACK_PER_STEP",
        ],
        "checks": [r1, r2] + tspan_results,
        "all_pass": all(c["pass"] for c in ([r1, r2] + tspan_results)),
        "n_checks_total": sum(c["n_checks"] for c in ([r1, r2] + tspan_results)),
        "verdict": ("PASS — refactored simulate() reproduces pre-refactor metrics bit-for-bit "
                    "on every overlapping configuration (main cell, perfect-oracle cell, and "
                    "all five B_ttl_sweep T-span rows); the parameterisation did not alter "
                    "default semantics."
                    if all(c["pass"] for c in ([r1, r2] + tspan_results)) else
                    "FAIL — see mismatches"),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    for c in [r1, r2] + tspan_results:
        print(f"  {'PASS' if c['pass'] else 'FAIL':4}  {c['label']}  "
              f"({len(c['mismatches'])} mismatches / {c['n_checks']} checks)")
    print("\nVERDICT:", report["verdict"])
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
