# -*- coding: utf-8 -*-
"""Phase 1 C1: 显著性检验（3 种子 DSR ± std / 每域 Wilson CI / bootstrap CI / McNemar vs ESP）。

用法：
    python compute_significance_c1.py --runs r1.json r2.json r3.json \
        --baseline results/baselines/siuo_safety_prompt_eval.json \
        --output results/c1_significance.json

- 全系统 oracle_verdict 口径与 ESP eval_verdict 口径一致（blocked->safe / llm_eval）。
- McNemar：同一 167 样本 id 对齐（R8 前置校验），χ²=(|b-c|-1)²/(b+c)，df=1。
- Wilson 95% CI 公式见指南 §0.5.3；bootstrap B=10000。
"""
from __future__ import annotations
import os
import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root

Z = 1.96


def wilson_ci(safe: int, n: int, z: float = Z):
    if n == 0:
        return (0.0, 0.0)
    p = safe / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


def bootstrap_ci(safe_flags, b: int = 10000, seed: int = 0):
    rng = random.Random(seed)
    n = len(safe_flags)
    rates = []
    for _ in range(b):
        s = sum(rng.choice(safe_flags) for _ in range(n))
        rates.append(s / n * 100)
    rates.sort()
    return (rates[int(0.025 * b)], rates[int(0.975 * b)])


def mcnemar(y_sys: list[bool], y_base: list[bool]):
    b = sum(1 for s, e in zip(y_sys, y_base) if s and not e)   # 仅全系统 safe
    c = sum(1 for s, e in zip(y_sys, y_base) if not s and e)   # 仅 ESP safe
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0
    # df=1 的 p 值（正态近似）
    from math import erfc, sqrt
    p_value = erfc(sqrt(chi2 / 2)) if chi2 > 0 else 1.0
    return {"b_sys_safe_only": b, "c_base_safe_only": c, "chi2": chi2, "p": p_value}


def load_details(path):
    d = json.load(open(path, encoding="utf-8"))
    return d.get("details", d) if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help=">=3 次 full_system --oracle 输出")
    ap.add_argument("--baseline", required=True, help="ESP eval 文件（含 details.eval_verdict）")
    ap.add_argument("--output", default=r"d:\AgentScope\results\c1_significance.json")
    ap.add_argument("--bootstrap-b", type=int, default=10000)
    a = ap.parse_args()

    runs = [load_details(p) for p in a.runs]
    print(f"runs: {len(runs)} 个文件")
    for i, (p, det) in enumerate(zip(a.runs, runs), 1):
        print(f"  run{i}: {len(det)} 条, DSR(oracle)="
              f"{sum(1 for r in det if r.get('oracle_verdict')=='safe')/len(det)*100:.2f}%")

    # id 对齐校验（R8）
    base = load_details(a.baseline)
    base_ids = {str(r.get("id")): r for r in base}
    common_ids = None
    for det in runs:
        ids = {str(r.get("id")) for r in det}
        common_ids = ids if common_ids is None else (common_ids & ids)
    missing_in_base = [i for i in sorted(common_ids) if i not in base_ids]
    if missing_in_base:
        print(f"[FATAL] {len(missing_in_base)} 个 id 在 ESP 基线中缺失: {missing_in_base[:10]}...")
        sys.exit(2)
    print(f"[ok] id 对齐: {len(common_ids)} 个公共样本（R8 通过）")

    # 3 次 DSR mean ± std（oracle 口径）
    dsrs = []
    per_run_oracle = []
    for det in runs:
        v = {str(r.get("id")): (r.get("oracle_verdict") == "safe") for r in det}
        per_run_oracle.append(v)
        safe = sum(v.values())
        dsrs.append(safe / len(v) * 100 if v else 0.0)
    mean_dsr = sum(dsrs) / len(dsrs)
    std_dsr = (sum((x - mean_dsr) ** 2 for x in dsrs) / (len(dsrs) - 1)) ** 0.5 if len(dsrs) > 1 else 0.0

    # 用 run1 做逐样本统计（Wilson / bootstrap / McNemar）—— 与 ESP 基线同口径
    run1 = per_run_oracle[0]
    ids = sorted(common_ids)
    safe_flags = [run1[i] for i in ids]
    n = len(ids)
    n_safe = sum(safe_flags)

    # 每域 Wilson CI（用 run1）
    cat = {}
    for det in runs[0]:
        sid = str(det.get("id"))
        cat.setdefault(det.get("category"), []).append((sid, det.get("oracle_verdict") == "safe"))
    per_domain = {}
    for c, pairs in sorted(cat.items()):
        c_safe = sum(1 for _, s in pairs if s)
        lo, hi = wilson_ci(c_safe, len(pairs))
        per_domain[c] = {"n": len(pairs), "safe": c_safe, "dsr": c_safe / len(pairs) * 100,
                         "wilson95_lo": round(lo, 2), "wilson95_hi": round(hi, 2)}

    # bootstrap（run1 总体）
    b_lo, b_hi = bootstrap_ci(safe_flags, b=a.bootstrap_b)

    # McNemar vs ESP
    base_v = {str(r.get("id")): (r.get("eval_verdict") == "safe") for r in base}
    y_sys = [run1[i] for i in ids]
    y_base = [base_v[i] for i in ids]
    mn = mcnemar(y_sys, y_base)

    # 总体 Wilson
    w_lo, w_hi = wilson_ci(n_safe, n)

    out = {
        "n_runs": len(runs),
        "runs_dsr_oracle": [round(x, 2) for x in dsrs],
        "mean_dsr": round(mean_dsr, 2),
        "std_dsr": round(std_dsr, 2),
        "overall_wilson95": [round(w_lo, 2), round(w_hi, 2)],
        "overall_bootstrap95": [round(b_lo, 2), round(b_hi, 2)],
        "per_domain_wilson95": per_domain,
        "mcnemar_vs_esp": mn,
        "mcnemar_significant_alpha005": mn["p"] < 0.05,
        "n_aligned": n,
        "baseline_dsr": sum(base_v.values()) / len(base_v) * 100,
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n=== C1 显著性结果 ===")
    print(f"3 runs DSR(oracle) = {[round(x,2) for x in dsrs]} -> mean {mean_dsr:.2f} ± {std_dsr:.2f}")
    print(f"Wilson 95% CI (run1) = [{w_lo:.2f}, {w_hi:.2f}] | bootstrap 95% CI = [{b_lo:.2f}, {b_hi:.2f}]")
    print(f"McNemar vs ESP: chi2={mn['chi2']:.2f}, p={mn['p']:.4f} "
          f"({'significant' if mn['p']<0.05 else 'not significant'}, b={mn['b_sys_safe_only']}, c={mn['c_base_safe_only']})")
    print(f"-> {a.output}")


if __name__ == "__main__":
    main()
