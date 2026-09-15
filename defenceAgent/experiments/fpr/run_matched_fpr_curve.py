# -*- coding: utf-8 -*-
"""Phase 1 (E6) B1/B2: 匹配 FPR 的 DSR/FPR 曲线（τ 扫描）。

设计（指南 §1.6）：
- 系统中唯一可连续调节安全/效用权衡的旋钮是 L2 相似度阈值 τ。
- 扫描 τ ∈ {0.70, 0.75, 0.80, 0.85, 0.90, 0.95}：
    * DSR(τ) on SIUO(167)：full_system 各 τ 的拦截率（dsr_block 口径，与论文一致）
    * FPR(τ) on benign 200 子集：full_system 各 τ 的误拦率
    * single_agent_control：无 L2 旋钮，单 operating point（SIUO 167 + benign 200）
- 输出 results/matched_fpr_curve.json，每个 (arm, τ) 组合独立缓存，重跑跳过已完成。

诚实局限（写进报告）：L3 Judge 裁决是二值的，全系统曲线主要由 L2 τ 扫描驱动，
控制/基线为单点；满足 R1-C1 "DSR/FPR curves" 并在 Response 中说明曲线形态。

成本：SIUO 臂 6×167 + 167，benign 臂 6×200 + 200。用 --dry-run 只算成本不下发。
"""
from __future__ import annotations
import os
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root

from defenceAgent.experiments.siuo.run_siuo_with_records import run_arm, set_oracle_model, DEFAULT_ORACLE_MODEL
import evaluate_fpr_real as fpr

TAUS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
BENIGN_PATH = r"d:\AgentScope\datasets\benign\benign_dataset.json"
BENIGN_SUBSET_SIZE = 200
BENIGN_SEED = 42
OUTPUT = r"d:\AgentScope\results\matched_fpr_curve.json"


def load_benign_subset(limit: int = BENIGN_SUBSET_SIZE, seed: int = BENIGN_SEED):
    """固定随机种子取 benign 子集（可复现）。"""
    samples = fpr.load_benign_dataset(BENIGN_PATH, limit=None)  # 953 有效
    rng = random.Random(seed)
    idx = rng.sample(range(len(samples)), min(limit, len(samples)))
    return [samples[i] for i in sorted(idx)]


async def benign_fpr_at_tau(tau: float, sample_ids_done, samples, output_dir: Path) -> float:
    """full_system(τ) 在 benign 子集上的误拦率（断点：jsonl 逐样本）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"benign_tau{tau:.2f}.jsonl"
    done = {}
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[str(r["id"])] = r
            except Exception:
                continue

    orch = None  # 惰性构建
    records = list(done.values())
    for i, s in enumerate(samples, 1):
        sid = str(s.get("id"))
        if sid in done:
            continue
        if orch is None:
            from defenceAgent.orchestrator import DefenceOrchestrator
            orch = DefenceOrchestrator(l2_threshold=tau, learning_enabled=False, cache_mode="off")
        user_msg = fpr.build_user_msg(s)
        if user_msg is None:
            rec = {"id": sid, "action": "error", "defense_source": "error", "is_false_positive": False}
        else:
            try:
                res = await fpr.evaluate_single_sample(orch, s)
                rec = {k: res.get(k) for k in ("id", "domain", "action", "defense_source", "is_false_positive")}
            except Exception as e:
                if fpr.is_balance_error(str(e)):
                    raise
                rec = {"id": sid, "domain": s.get("domain"), "action": "error",
                       "defense_source": "error", "is_false_positive": False}
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        records.append(rec)
        if i % 50 == 0:
            print(f"  [benign τ={tau}] {i}/{len(samples)}")

    valid = [r for r in records if r.get("action") != "error"]
    blocked = sum(1 for r in valid if r.get("action") == "block")
    return (blocked / len(valid) * 100) if valid else 0.0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="*", default=TAUS, help="覆盖默认 τ 扫描集")
    ap.add_argument("--siuo-limit", type=int, default=None, help="SIUO 条数上限（pilot 用）")
    ap.add_argument("--benign-limit", type=int, default=BENIGN_SUBSET_SIZE)
    ap.add_argument("--output", default=OUTPUT)
    ap.add_argument("--oracle-model", default=DEFAULT_ORACLE_MODEL)
    ap.add_argument("--dry-run", action="store_true", help="只打印成本估算不下发")
    ap.add_argument("--skip-single-agent", action="store_true", help="跳过 single_agent_control 单点（已在其他进程跑过）")
    a = ap.parse_args()

    if a.dry_run:
        n_siuo = (a.siuo_limit or 167)
        n_benign = min(a.benign_limit, 953)
        runs = (len(a.taus) + 1) * n_siuo + (len(a.taus) + 1) * n_benign
        print(f"[dry-run] τ={a.taus}")
        print(f"  SIUO runs : {len(a.taus)+1} arm/τ × {n_siuo} = {(len(a.taus)+1)*n_siuo}")
        print(f"  benign runs: {len(a.taus)+1} arm/τ × {n_benign} = {(len(a.taus)+1)*n_benign}")
        print(f"  总系统运行 ≈ {runs} 次（每次约 3-5 次模型调用 + oracle）")
        return

    set_oracle_model(a.oracle_model)
    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    benign_dir = out_path.parent / "matched_fpr"

    # 已有结果
    curve = {}
    if out_path.exists():
        curve = json.load(open(out_path, encoding="utf-8"))
        print(f"[resume] 已有结果: {list(curve.keys())}")

    siuo_data = None  # 惰性加载

    # 1) full_system τ 扫描
    for tau in a.taus:
        key = f"full_system@tau{tau:.2f}"
        if key in curve and curve[key].get("dsr") is not None and curve[key].get("fpr") is not None:
            print(f"[skip] {key} 已存在")
            continue
        print(f"=== {key}: SIUO DSR ===")
        res = await run_arm(
            arm="full_system", l2_threshold=tau, limit=a.siuo_limit,
            output=str(out_path.parent / f"siuo_tau{tau:.2f}.json"),
            oracle=False, oracle_model=a.oracle_model, resume=True,
            run_id=f"tau{tau:.2f}", retry_errors=False,
        )
        dsr = res["summary"]["dsr_block"]
        print(f"=== {key}: benign FPR (n={a.benign_limit}) ===")
        benign = load_benign_subset(a.benign_limit)
        fpr_val = await benign_fpr_at_tau(tau, None, benign, benign_dir)
        curve[key] = {"arm": "full_system", "tau": tau, "dsr": round(dsr, 2), "fpr": round(fpr_val, 2)}
        json.dump(curve, open(out_path, "w", encoding="utf-8"), indent=2)
        print(f"[saved] {key}: DSR={dsr:.2f}% FPR={fpr_val:.2f}%")

    # 2) single_agent_control 单点
    key = "single_agent_control@point"
    if a.skip_single_agent:
        print("[skip] single_agent_control 单点（--skip-single-agent）")
    elif key not in curve or curve[key].get("dsr") is None or curve[key].get("fpr") is None:
        print(f"=== {key}: SIUO DSR ===")
        res = await run_arm(
            arm="single_agent_control", l2_threshold=0.85, limit=a.siuo_limit,
            output=str(out_path.parent / "single_agent_control.json"),
            oracle=False, oracle_model=a.oracle_model, resume=True,
            run_id="single_agent", retry_errors=False,
        )
        dsr = res["summary"]["dsr_block"]
        print(f"=== {key}: benign FPR (n={a.benign_limit}) ===")
        benign = load_benign_subset(a.benign_limit)
        # 单 agent 不走 L2：benign 全部过 L3 单 agent
        from defenceAgent.orchestrator_ablation import SingleAgentControlOrchestrator
        orch = SingleAgentControlOrchestrator()
        blocked = 0
        valid = 0
        fpr_file = benign_dir / "benign_single_agent.jsonl"
        done_ids = set()
        if fpr_file.exists():
            for line in fpr_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        done_ids.add(str(r["id"]))
                    except Exception:
                        pass
        for i, s in enumerate(benign, 1):
            sid = str(s.get("id"))
            if sid in done_ids:
                continue
            user_msg = fpr.build_user_msg(s)
            try:
                out = await orch.run(input_msg=user_msg)
                v = out.get("verdict", "error")
                action = "block" if v in ("unsafe", "conditional_unsafe") else "allow"
            except Exception as e:
                if fpr.is_balance_error(str(e)):
                    raise
                action = "error"
            with open(fpr_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": sid, "action": action}, ensure_ascii=False) + "\n")
        recs = []
        for line in fpr_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
        valid = [r for r in recs if r.get("action") != "error"]
        blocked = sum(1 for r in valid if r.get("action") == "block")
        fpr_val = (blocked / len(valid) * 100) if valid else 0.0
        curve[key] = {"arm": "single_agent_control", "tau": None, "dsr": round(dsr, 2), "fpr": round(fpr_val, 2)}
        json.dump(curve, open(out_path, "w", encoding="utf-8"), indent=2)
        print(f"[saved] {key}: DSR={dsr:.2f}% FPR={fpr_val:.2f}%")

    print("\n=== 完成 ===")
    for k, v in curve.items():
        print(f"  {k}: DSR={v['dsr']}% FPR={v['fpr']}%")
    print(f"-> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
