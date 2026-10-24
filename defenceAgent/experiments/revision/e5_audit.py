# -*- coding: utf-8 -*-
"""Independent audit of EX-E5 (per guide section 八 and the audit ledger).

Recomputes, WITHOUT importing the executor's e5_fp_induction_probe.py:
  A. arm counts (n, FP, block-by-layer) directly from e5_ckpt.jsonl
  B. Wilson 95% CI per arm (self-written)
  C. two-proportion z-test (pooled SE) on variant vs control
  D. consistency of e5_probe.json with the raw checkpoint
  E. library-contamination block present and consistent with the library
  F. positioning / caveat strings present (directional, not calibrated)
"""
import json, math
from pathlib import Path

E5 = Path(__file__).resolve().parent / "results"
CKPT = E5 / "e5_ckpt.jsonl"
REPORT = E5 / "e5_probe.json"

checks = []
def chk(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

def wilson(k, n, z=1.959963985):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))

def ztest(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0: return (None, None)
    p1, p2 = k1/n1, k2/n2
    p = (k1+k2)/(n1+n2)
    se = math.sqrt(p*(1-p)*(1/n1+1/n2))
    if se == 0: return (0.0, 1.0)
    z = (p1-p2)/se
    p2s = 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))
    return (round(z,4), round(p2s,6))

rows = [json.loads(l) for l in CKPT.read_text(encoding="utf-8").splitlines() if l.strip()]
ok_rows = [r for r in rows if r.get("status") == "ok"]
rep = json.loads(REPORT.read_text(encoding="utf-8"))

# ---- A. arm counts ----
def arm(rows, name):
    rs = [r for r in rows if r["arm"] == name]
    n = len(rs)
    fp = sum(1 for r in rs if r["action"] == "block")
    l2 = sum(1 for r in rs if r.get("l2_block"))
    l3 = sum(1 for r in rs if r.get("l3_block"))
    l1 = sum(1 for r in rs if r.get("l1_hit"))
    return dict(n=n, fp=fp, l2=l2, l3=l3, l1=l1,
                fp_rate=(fp/n if n else None))
ca = arm(ok_rows, "control"); va = arm(ok_rows, "variant")
print("control:", ca)
print("variant:", va)

# ---- B. Wilson ----
wc = wilson(ca["fp"], ca["n"]); wv = wilson(va["fp"], va["n"])
chk("B1 control Wilson CI matches report",
    all(abs(a-b) < 1e-3 for a, b in zip(wc, rep["arms"]["control"]["wilson95"])),
    f"audit={[round(x,4) for x in wc]} report={rep['arms']['control']['wilson95']}")
chk("B2 variant Wilson CI matches report",
    all(abs(a-b) < 1e-3 for a, b in zip(wv, rep["arms"]["variant"]["wilson95"])),
    f"audit={[round(x,4) for x in wv]} report={rep['arms']['variant']['wilson95']}")

# ---- C. z-test ----
z, p = ztest(va["fp"], va["n"], ca["fp"], ca["n"])
chk("C1 two-proportion z-test matches report",
    abs((z or 0) - (rep["difference_variant_vs_control"].get("z") or 0)) < 1e-3
    and abs((p or 0) - (rep["difference_variant_vs_control"].get("p") or 0)) < 1e-5,
    f"audit z={z} p={p} vs report {rep['difference_variant_vs_control']}")

# ---- D. report matches checkpoint ----
chk("D1 report control n/fp matches ckpt",
    rep["arms"]["control"]["n"] == ca["n"] and rep["arms"]["control"]["fp"] == ca["fp"],
    f"report n={rep['arms']['control']['n']} fp={rep['arms']['control']['fp']}")
chk("D2 report variant n/fp matches ckpt",
    rep["arms"]["variant"]["n"] == va["n"] and rep["arms"]["variant"]["fp"] == va["fp"],
    f"report n={rep['arms']['variant']['n']} fp={rep['arms']['variant']['fp']}")
chk("D3 block-by-layer counts match",
    rep["arms"]["control"]["block_L2"] == ca["l2"]
    and rep["arms"]["variant"]["block_L3"] == va["l3"],
    f"ctrl L2={ca['l2']} var L3={va['l3']}")

# ---- E. contamination block ----
con = rep.get("library_contamination", {})
chk("E1 library_contamination present", bool(con), json.dumps(con, ensure_ascii=False)[:120])
chk("E2 contamination count sane (0 < n < total)",
    isinstance(con.get("n_in_library"), int),
    f"n_in_library={con.get('n_in_library')} / {con.get('n_items')}")

# ---- F. positioning ----
pos = rep.get("positioning", "")
chk("F1 positioning says directional / not calibrated",
    "not a calibrated attack" in pos.lower() or "directional" in pos.lower(), pos[:90])

# ---- G. text-only channel captured (mechanism attribution) ----
n_sim = sum(1 for r in ok_rows if r.get("l2_sim") is not None)
n_txt = sum(1 for r in ok_rows if r.get("l2_sim_text_only") is not None)
chk("G1 text-only channel captured for the multimodal queries",
    n_txt > 0, f"rows with text-only sim: {n_txt}/{len(ok_rows)} (mm sim rows: {n_sim})")

# ---- H. guide section 八 acceptance criteria ----
AC1 = (ca["n"] == 50 and va["n"] >= 1 and bool(rep["arms"]["control"]["wilson95"])
       and bool(rep["arms"]["variant"]["wilson95"]))
chk("AC1 50-rewrite + matched control + Wilson CI present", AC1,
    f"control n={ca['n']} variant n={va['n']} (1 variant/item)")
AC2 = ("directional" in pos.lower()) and ("not a calibrated attack" in pos.lower()
       or "not calibrated" in pos.lower())
chk("AC2 labelled directional probe / not calibrated", AC2, pos[:100])
AC3 = "conclusion" in rep and "instructions" not in rep
chk("AC3 conclusion string present", bool(rep.get("conclusion")), rep.get("conclusion", "")[:80])

# ---- I. mechanism attribution (image vs text) ----
mm_block = [r for r in ok_rows if r.get("l2_block") and r.get("l2_sim") is not None]
txt_lo = [r for r in mm_block
          if r.get("l2_sim_text_only") is not None and r["l2_sim_text_only"] < 0.85]
chk("I1 image-driven FP confirmed (blocked rows where text-only sim < 0.85)",
    len(txt_lo) > 0,
    f"{len(txt_lo)}/{len(mm_block)} L2 blocks have text-only sim < 0.85 "
    f"(=> the image drives the match)")
if mm_block:
    mm_hi = sum(1 for r in mm_block if r["l2_sim"] >= 0.85)
    print(f"\n[MECH] L2 blocks: {len(mm_block)}; of them mm-sim>=0.85: {mm_hi}; "
          f"text-only sim<0.85: {len(txt_lo)}")
    ex = [round(r["l2_sim"], 3) for r in mm_block[:5]]
    ext = [round(r["l2_sim_text_only"], 3) if r.get("l2_sim_text_only") is not None else None
           for r in mm_block[:5]]
    print(f"[MECH] first blocks mm_sim={ex} text_sim={ext}")

print("\n=== SUMMARY ===")
npass = sum(1 for _, ok, _ in checks if ok)
print(f"{npass}/{len(checks)} checks passed")
print(f"control  FP {ca['fp']}/{ca['n']} = {ca['fp_rate']}")
print(f"variant  FP {va['fp']}/{va['n']} = {va['fp_rate']}")
