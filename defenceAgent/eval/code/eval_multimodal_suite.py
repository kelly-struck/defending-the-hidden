from __future__ import annotations
"""
在同一批 MM-SafetyBench 多模态样本上，统一运行三种设置：
- FigStep 基线（defenceAgent.eval_figstep.eval_figstep）
- 无防御（defenceAgent.eval_compare.eval_without_defence）
- 带防御（defenceAgent.eval_compare.eval_with_defence）

从而避免数据切分不一致带来的不可比问题。输出一个汇总 JSON 报告。
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from .dataset_utils import load_mmsafetybench_rows
from .eval_figstep import eval_figstep as run_figstep
from .eval_compare import eval_without_defence as run_no_defence
from .eval_compare import eval_with_defence as run_with_defence


def main():
    parser = argparse.ArgumentParser(description="统一多模态评测套件：FigStep 基线 / 无防御 / 带防御")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "statics" / "dataset"),
    )
    parser.add_argument("--branch", type=str, default="EconomicHarm")
    parser.add_argument(
        "--split",
        type=str,
        default="SD.parquet",
        help="可选：SD.parquet | SD_TYPO.parquet | TYPO.parquet | Text_only.parquet",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--pro", action="store_true", help="启用 FigStep-Pro（切图/多图）")
    parser.add_argument("--no-classifier", action="store_true", help="关闭模型安全分类器（仅启发式）")

    args = parser.parse_args()

    # 统一载入相同 rows
    rows = load_mmsafetybench_rows(
        args.dataset_dir, args.branch, args.split.replace(".parquet", ""), None
    )
    if args.limit is not None:
        rows = rows[: args.limit]

    # 逐项评测
    figstep = run_figstep(rows, limit=None, pro=args.pro, use_classifier=(not args.no_classifier))
    no_def = run_no_defence(rows, limit=None, use_classifier=(not args.no_classifier))
    with_def = run_with_defence(rows, limit=None)

    parquet_path = Path(args.dataset_dir) / "data" / args.branch / args.split
    report = {
        "dataset": str(parquet_path),
        "limit": len(rows),
        "figstep_baseline": figstep,
        "without_defence": no_def,
        "with_defence": with_def,
        "delta_attack_success_rate": {
            "no_def_vs_with_def": no_def["attack_success_rate"] - with_def["attack_success_rate"],
            "figstep_vs_with_def": figstep["attack_success_rate"] - with_def["attack_success_rate"],
            "figstep_vs_no_def": figstep["attack_success_rate"] - no_def["attack_success_rate"],
        },
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
