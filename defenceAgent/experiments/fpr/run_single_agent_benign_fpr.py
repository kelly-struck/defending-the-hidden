# -*- coding: utf-8 -*-
"""Phase 1 B1/B2: single_agent_control 单点的 benign FPR（200 子集，断点 jsonl）。"""
from __future__ import annotations
import os
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root

import evaluate_fpr_real as fpr
from defenceAgent.orchestrator_ablation import SingleAgentControlOrchestrator

BENIGN_PATH = r"d:\AgentScope\datasets\benign\benign_dataset.json"
SUBSET_SIZE = 200
SEED = 42
OUT = Path(r"d:\AgentScope\results\phase1\matched_fpr\benign_single_agent.jsonl")


def load_subset():
    samples = fpr.load_benign_dataset(BENIGN_PATH, limit=None)
    rng = random.Random(SEED)
    idx = rng.sample(range(len(samples)), min(SUBSET_SIZE, len(samples)))
    return [samples[i] for i in sorted(idx)]


async def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    done[str(r["id"])] = r
                except Exception:
                    pass
    print(f"[resume] {len(done)}/{SUBSET_SIZE} 已完成")

    orch = SingleAgentControlOrchestrator()
    samples = load_subset()
    for i, s in enumerate(samples, 1):
        sid = str(s.get("id"))
        if sid in done:
            continue
        user_msg = fpr.build_user_msg(s)
        try:
            out = await orch.run(input_msg=user_msg)
            v = out.get("verdict", "error")
            action = "block" if v in ("unsafe", "conditional_unsafe") else "allow"
        except Exception as e:
            if fpr.is_balance_error(str(e)):
                print(f"[FATAL] 余额不足，中止于 {len(done)}/{SUBSET_SIZE}")
                sys.exit(3)
            action = "error"
        rec = {"id": sid, "domain": s.get("domain"), "action": action}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done[sid] = rec
        if i % 50 == 0:
            print(f"[benign single_agent] {i}/{SUBSET_SIZE}")

    valid = [r for r in done.values() if r.get("action") != "error"]
    blocked = sum(1 for r in valid if r.get("action") == "block")
    fpr_val = blocked / len(valid) * 100 if valid else 0.0
    print(f"FPR = {fpr_val:.2f}% (blocked={blocked}/{len(valid)})")
    json.dump({"arm": "single_agent_control", "fpr": round(fpr_val, 2), "blocked": blocked, "n": len(valid)},
              open(str(OUT).replace(".jsonl", "_summary.json"), "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    asyncio.run(main())
