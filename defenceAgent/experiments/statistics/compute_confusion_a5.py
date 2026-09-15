# -*- coding: utf-8 -*-
"""Phase 1 A5: T6 对称规则 vs 无防御 混淆矩阵诊断。

- 行 = 无防御 oracle 判（safe/unsafe）：oracle 判受害模型直答安全 = 无防御本可拦
- 列 = 对称规则判（allow/block）
- 重点单元格 a = (无防御 oracle safe, 对称 allow)：对称规则删除了无防御已正确
  拦截的样本 -> 若显著，证实"框架降低安全性"（R2-C19）。
- 导出 >=5 个失败样本（symmetric=allow 且应为 unsafe）的完整 Judge 输出。
"""
from __future__ import annotations
import os
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root


def load_details(path):
    d = json.load(open(path, encoding="utf-8"))
    return d.get("details", d) if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symmetric", required=True, help="symmetric arm 输出（含 details 与 oracle_verdict）")
    ap.add_argument("--nodefense", required=True, help="no_defense arm 输出（含 details 与 oracle_verdict）")
    ap.add_argument("--output", default=r"d:\AgentScope\results\a5_confusion.json")
    ap.add_argument("--max-failures", type=int, default=8)
    a = ap.parse_args()

    sym = {str(r.get("id")): r for r in load_details(a.symmetric)}
    nd = {str(r.get("id")): r for r in load_details(a.nodefense)}

    ids = sorted(set(sym.keys()) & set(nd.keys()))
    matrix = {"a_nd_safe_sym_allow": 0, "b_nd_safe_sym_block": 0,
              "c_nd_unsafe_sym_allow": 0, "d_nd_unsafe_sym_block": 0}
    failures = []  # symmetric=allow 且应为 unsafe（SIUO 全 unsafe）：重点是 oracle 也判 unsafe 的
    nd_oracle_safe_total = 0
    for sid in ids:
        s = sym[sid]
        n = nd[sid]
        sym_action = s.get("final_action")  # allow/block
        nd_oracle_safe = (n.get("oracle_verdict") == "safe")
        if nd_oracle_safe:
            nd_oracle_safe_total += 1
        key = ("a_nd_safe_sym_allow" if sym_action == "allow" else "b_nd_safe_sym_block") if nd_oracle_safe \
            else ("c_nd_unsafe_sym_allow" if sym_action == "allow" else "d_nd_unsafe_sym_block")
        matrix[key] += 1
        if sym_action == "allow":
            failures.append({
                "id": s.get("id"), "category": s.get("category"),
                "question": s.get("question", "")[:150],
                "symmetric_verdict": s.get("verdict"),
                "symmetric_judge_output": (s.get("judge_reason") or "")[:1500],
                "nd_oracle_verdict": n.get("oracle_verdict"),
                "nd_response_preview": (n.get("response") or "")[:200],
            })

    n_total = len(ids)
    a_, b_, c_, d_ = matrix.values()
    print(f"=== 混淆矩阵 (n={n_total}) ===")
    print(f"                   | Sym allow | Sym block |")
    print(f"Undef oracle safe  | {a_:>9} | {b_:>9} |")
    print(f"Undef oracle unsafe| {c_:>9} | {d_:>9} |")
    print(f"\n重点单元格 a (undef safe & sym allow) = {a_} "
          f"({a_/n_total*100:.1f}% of {n_total}) —— 对称规则删除了无防御已正确拦截的样本")
    print(f"对称规则相对无防御的净增益 d = {d_} (拦下无防御漏掉的)")

    # 失败样本：对称放行且确认有害
    confirmed = [f for f in failures if f["nd_oracle_verdict"] == "unsafe"]
    print(f"\n失败样本（sym allow 且 oracle 判 unsafe）: {len(confirmed)} 条，导出前 {a.max_failures} 条")
    for f in confirmed[:a.max_failures]:
        print(f"  [{f['id']}] {f['category']}: {f['question'][:60]}...")
        print(f"     judge: {f['symmetric_judge_output'][:200]}")

    out = {
        "n": n_total,
        "confusion": matrix,
        "undef_oracle_safe_total": nd_oracle_safe_total,
        "focus_cell_a": a_,
        "focus_cell_a_pct": round(a_ / n_total * 100, 2) if n_total else 0.0,
        "failures_export": confirmed[:a.max_failures],
        "interpretation": (
            "a>0 显著 => 对称规则删除了无防御已正确拦截的样本（框架降低安全性，Lim.2 关联发现）；"
            "若 a≈0 且 c 大 => 对称规则本身漏检；若 verdict 反转异常 => 检查误实现"
        ),
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"-> {a.output}")


if __name__ == "__main__":
    main()
