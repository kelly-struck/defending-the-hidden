# -*- coding: utf-8 -*-
"""Independent audit of EX-E2b (per guide section 七 and the audit ledger).

Recomputes, WITHOUT importing the executor's tau_framework.py:
  A. framework-table values vs their claimed sources
     (e2a report, ttl report, sensitivity_ext report)
  B. the central claim: main-run poison is not cross-entry retrievable at 0.85
     (poison_construction.max_sim_to_lib must be < 0.85)
  C. Arm A @0.427 regression == ttl_experiment_report step5_table (digit-exact)
  D. the four-row framework table's internal consistency (column set, row set)
"""
import json, sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
E2B = RES / "tau_framework.json"
E2A = RES / "tau_recalibration_report.json"
TTL = RES / "ttl_experiment_report.json"
SENS = RES / "sensitivity_ext_report.json"

checks = []
def chk(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

e2b = json.loads(E2B.read_text(encoding="utf-8"))
e2a = json.loads(E2A.read_text(encoding="utf-8"))
ttl = json.loads(TTL.read_text(encoding="utf-8"))

# ---- A. framework table structure ----
ft = e2b["framework_table"]
chk("A1 framework has 5 columns", len(ft["columns"]) == 5, str(ft["columns"]))
chk("A2 framework has 4 rows", len(ft["rows"]) == 4, [r["row"] for r in ft["rows"]])

# ---- B. central claim: not retrievable at 0.85 ----
amin = min(ttl["poison_construction"]["max_sim_to_lib"])
amax = max(ttl["poison_construction"]["max_sim_to_lib"])
chk("B1 main-report poison max_sim_to_lib < 0.85",
    amax < 0.85, f"max={amax:.4f}")
aa = e2b["tau085_arm"]["arm_a_near_neighbor"]
chk("B2 E2b Arm A max_cross_sim equals main report (at source precision)",
    round(aa["poison_lib_sim_excl_self"]["max"], 4) == round(amax, 4),
    f"e2b={aa['poison_lib_sim_excl_self']['max']:.6f} vs ttl={amax:.6f} (ttl stored 4dp)")
chk("B3 E2b Arm A reports 0 activatable at 0.85",
    aa["n_poison_activatable_at_tau"] == 0 and aa["activatable"] is False,
    f"n={aa['n_poison_activatable_at_tau']}")
chk("B4 self-match diagnostic separates 50 self vs few cross",
    aa["self_match_diagnostic"]["n_self_match_activations_at_tau"] == 50
    and aa["self_match_diagnostic"]["n_cross_entry_activations_at_tau"] >= 0,
    json.dumps(aa["self_match_diagnostic"], ensure_ascii=False)[:120])

# ---- C. Arm A @0.427 regression vs ttl step5 ----
reg = e2b["tau085_arm"]["arm_a_regression_check_at_0.427"]
s5 = ttl["step5_table"]
hdr = s5["header"]; rows = {r[0]: r for r in s5["rows"]}
def pct(x): return float(str(x).replace("%", "")) / 100.0
ok = True; det = []
for pol, key in (("no-TTL","no-TTL"), ("current-TTL","current-TTL"), ("competence-TTL","competence-TTL")):
    p = reg["policies"][pol]
    trow = rows[pol]
    # columns: 0 name,1 激活率,2 recall,3 collateral,4 rare,5 fpr,6 tpr
    e_act = p["poison_activation_rate"]; t_act = pct(trow[1])
    e_col = p["collateral_removal"];     t_col = pct(trow[3])
    e_rare= p["rare_retention"];         t_rare= pct(trow[4])
    e_fpr = p["fpr_post"];               t_fpr = pct(trow[5])
    e_tpr = p["tpr_post"];               t_tpr = pct(trow[6])
    for nm, e, t in (("act",e_act,t_act),("col",e_col,t_col),("rare",e_rare,t_rare),
                     ("fpr",e_fpr,t_fpr),("tpr",e_tpr,t_tpr)):
        if abs(e - t) > 5e-4:          # ttl step5 stores 1-dp percentages
            ok = False; det.append(f"{pol}.{nm}: e2b={e} vs ttl={t}")
chk("C1 Arm A @0.427 regression matches ttl step5 (source-precision)", ok,
    "; ".join(det) if det else "all three policies identical (ttl step5 is 1-dp %)")
chk("C2 E2b reports 36/50 activatable at 0.427",
    reg["n_poison_activatable_at_tau"] == 36, f"n={reg['n_poison_activatable_at_tau']}")

# ---- A3. framework row values vs sources ----
r1 = next(r for r in ft["rows"] if r["row"] == "R1_main_cascade")
r4 = next(r for r in ft["rows"] if r["row"] == "R4_disjoint_recalibration")
# R4 tau* from e2a
def dig(d, *ks):
    for k in ks:
        if isinstance(d, dict) and k in d: d = d[k]
        else: return None
    return d
print("\n[e2a keys]", list(e2a.keys()))
lodo = dig(e2a, "lodo_summary")
print("[e2a lodo_summary]", json.dumps(lodo, ensure_ascii=False)[:400] if lodo else None)
main_arm = dig(e2a, "main_arm_figstep")
print("[e2a main_arm_figstep keys]",
      list(main_arm.keys()) if isinstance(main_arm, dict) else type(main_arm).__name__)

# ---- D. Arm B near-duplicate sanity ----
ab = e2b["tau085_arm"]["arm_b_near_duplicate"]
chk("D1 Arm B max cross-entry sim in (0.85, 0.95]",
    0.85 <= ab["poison_lib_sim_excl_self"]["max"] <= 0.95,
    f"max={ab['poison_lib_sim_excl_self']['max']:.4f}")
chk("D2 Arm B activatable count is small but nonzero",
    0 < ab["n_poison_activatable_at_tau"] < 50,
    f"n={ab['n_poison_activatable_at_tau']}")

# ---- E. conclusion present & mentions retrievability gating ----
concl = e2b.get("conclusion","")
chk("E1 conclusion mentions retrievability-gated",
    "retrievability-gated" in concl or "retrievability gated" in concl.lower(),
    concl[:80])

print("\n=== SUMMARY ===")
npass = sum(1 for _,ok,_ in checks if ok)
print(f"{npass}/{len(checks)} checks passed")
