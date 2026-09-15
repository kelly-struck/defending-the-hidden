# -*- coding: utf-8 -*-
"""Phase 1 (E5) 无防御 arm：受害模型直答 SIUO，由独立 oracle 判安全与否。

用于 A5（对称规则 vs 无防御混淆矩阵）与 B1/B2（曲线基准 arm）。
实现复用 run_siuo_with_records.run_arm（同一套断点/错误处理/oracle 逻辑）。

示例：
    python run_no_defense.py --output results/a5_nodefense_with_oracle.json --oracle
"""
from __future__ import annotations
import os
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root

from defenceAgent.experiments.siuo.run_siuo_with_records import run_arm, DEFAULT_ORACLE_MODEL, DEFAULT_TARGET_MODEL


def main():
    ap = argparse.ArgumentParser(description="Phase1 E5: 无防御（受害模型直答）arm")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--oracle-model", default=DEFAULT_ORACLE_MODEL)
    ap.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument("--run-id", default="no_defense")
    a = ap.parse_args()

    asyncio.run(run_arm(
        arm="no_defense", l2_threshold=0.85, limit=a.limit,
        output=a.output, oracle=a.oracle, oracle_model=a.oracle_model,
        resume=a.resume, run_id=a.run_id, retry_errors=False,
        target_model=a.target_model,
    ))
    sys.exit(0)


if __name__ == "__main__":
    main()
