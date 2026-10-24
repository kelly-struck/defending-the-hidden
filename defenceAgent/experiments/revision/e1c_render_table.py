# -*- coding: utf-8 -*-
"""Render benchmark_full_report.json -> benchmark_full_table.md (markdown tables).

Kept separate from the eval script so the table can be re-rendered without
re-running any API calls.
"""
from __future__ import annotations
import json
from pathlib import Path

E1C = Path(__file__).resolve().parent / "e1c" / "results" / "benchmark_full_report.json"
E2A = Path(__file__).resolve().parent / "e2a" / "results" / "tau_recalibration_report.json"
OUT = Path(__file__).resolve().parent / "benchmark_full_table.md"


def pct(x: float) -> str:
    return f"{x*100:.2f}%"


def ci(d: dict) -> str:
    lo, hi = d["wilson95"]
    return f"[{lo*100:.2f}, {hi*100:.2f}]"


def main():
    lines = []
    if E1C.exists():
        rep = json.loads(E1C.read_text(encoding="utf-8"))
        lines.append("# EX-E1-③ | HarmBench / MM-SafetyBench 完整报告（L3-only）\n")
        lines.append(f"- 抽样种子：{rep['config']['seed']}；regime：**{rep['config']['regime']}**；"
                     f"orchestrator：`{rep['config']['orchestrator']}`（learning_enabled=False）\n")
        lines.append(f"- L3-only 理由：{rep['l3_only_rationale']}\n")
        lines.append("## 主表（ASR = allow/N，含 Wilson 95% CI）\n")
        lines.append("| Benchmark | N | ASR | Wilson 95% CI | Refusal | Avg Tokens |")
        lines.append("|---|---|---|---|---|---|")
        for b, r in rep["results"].items():
            lines.append(f"| {b} | {r['N']} | {pct(r['asr']['p'])} | {ci(r['asr'])} | "
                         f"{pct(r['refusal_rate']['p'])} | {r['avg_tokens']:.0f} |")
        lines.append("")
        for b, r in rep["results"].items():
            lines.append(f"## {b} 逐类（N / ASR / CI）\n")
            lines.append("| Category | N | ASR | Wilson 95% CI |")
            lines.append("|---|---|---|---|")
            for c, d in r["per_category"].items():
                lines.append(f"| {c} | {d['N']} | {pct(d['asr']['p'])} | {ci(d['asr'])} |")
            lines.append("")
            if len(r.get("per_config", {})) > 1:
                lines.append(f"## {b} 按 config\n")
                lines.append("| Config | N | ASR | Wilson 95% CI |")
                lines.append("|---|---|---|---|")
                for c, d in r["per_config"].items():
                    lines.append(f"| {c} | {d['N']} | {pct(d['asr']['p'])} | {ci(d['asr'])} |")
                lines.append("")
    else:
        lines.append("(!) benchmark_full_report.json 未找到\n")

    if E2A.exists():
        rep = json.loads(E2A.read_text(encoding="utf-8"))
        m = rep["main_arm_figstep"]
        lines.append("\n# EX-E2a | FigStep disjoint 阈值重标定 + LODO\n")
        lines.append(f"- disjoint 证据：FigStep 不在库内（库来源 = {rep['disjointness_evidence']['library_sources']}）\n")
        lines.append(f"- 主臂：{m['attack_set']}，query={m['query_mode']}，n_attack={m['n_attack']}，n_benign={m['n_benign']}\n")
        lines.append("## 主臂结果\n")
        lines.append("| 指标 | 值 |")
        lines.append("|---|---|")
        lines.append(f"| attack top-1 sim 中位/最大 | {m['attack_sim_median']:.3f} / {m['attack_sim_max']:.3f} |")
        lines.append(f"| benign top-1 sim 中位/最大 | {m['benign_sim_median']:.3f} / {m['benign_sim_max']:.3f} |")
        lines.append(f"| τ=0.85 攻击可检索率(TPR) | {pct(m['tpr_at_prod_tau_0.85'])} |")
        lines.append(f"| τ=0.85 良性 FPR | {pct(m['fpr_at_prod_tau_0.85'])} |")
        lines.append(f"| τ*（precision-first, FPR≤1%） | {m['tau_star']['precision_first']['tau']} "
                     f"(TPR={pct(m['tau_star']['precision_first']['tpr'])}, FPR={pct(m['tau_star']['precision_first']['fpr'])}) |")
        lines.append(f"| τ*（Youden） | {m['tau_star']['youden']['tau']} "
                     f"(J={m['tau_star']['youden']['youden_j']:.3f}) |")
        lines.append("")
        lines.append("## τ 扫描曲线\n")
        lines.append("| τ | TPR | FPR | Youden J |")
        lines.append("|---|---|---|---|")
        for row in m["curve"]:
            lines.append(f"| {row['tau']:.2f} | {pct(row['tpr'])} | {pct(row['fpr'])} | {row['youden_j']:.3f} |")
        if rep.get("lodo_folds"):
            lines.append("\n## LODO（leave-one-MM-SafetyBench-category-out）\n")
            lines.append("| 抽出类别 | 库大小 | 探针 N | 攻击 sim 中位/最大 | τ*(precision-first) | τ*(Youden) |")
            lines.append("|---|---|---|---|---|---|")
            for f in rep["lodo_folds"]:
                lines.append(f"| {f['held_out_category']} | {f['library_size']} | {f['n_probe']} | "
                             f"{f['attack_sim_median']:.3f}/{f['attack_sim_max']:.3f} | "
                             f"{f['tau_star_precision_first']} | {f['tau_star_youden']} |")
        lines.append(f"\n**结论**：{rep['conclusion']}\n")
        lines.append(f"> {rep['d5a_note']}\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
