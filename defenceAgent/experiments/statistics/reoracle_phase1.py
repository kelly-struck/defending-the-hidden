# -*- coding: utf-8 -*-
"""一次性：用 qwen-plus（evaluate_baselines 同口径）重判 allow 样本 oracle。

原因：qwen2.5-7b-instruct 全系列 AccessDenied，此前 allow 样本 oracle 全为 unknown（无效）。
allow 样本的 response 已保存在 jsonl/JSON，无需重跑辩论。

对每个输入 JSON：
- blocked 样本保持 pre_blocked=safe
- allow 样本：heuristic fast path -> call_eval_model(qwen-plus)
- 更新 summary 并写回 JSON + 同步 jsonl
"""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root
import defenceAgent.experiments.baselines.evaluate_baselines as eb
from defenceAgent.experiments.siuo.run_siuo_with_records import set_oracle_model, _call_oracle_with_retry

ORACLE_MODEL = "qwen-plus"


def main():
    set_oracle_model(ORACLE_MODEL)
    for path in sys.argv[1:]:
        p = Path(path)
        data = json.load(open(p, encoding="utf-8"))
        details = data["details"]
        n_allow = 0
        n_heur = 0
        n_llm = 0
        for r in details:
            if r.get("final_action") == "allow":
                n_allow += 1
                if eb.is_refusal_heuristic(r.get("response", "")):
                    r["oracle_verdict"] = "safe"
                    r["oracle_method"] = "heuristic"
                    r["oracle_reason"] = "heuristic_refusal_detected"
                    n_heur += 1
                else:
                    ev = _call_oracle_with_retry(eb.call_eval_model, r.get("question", ""), r.get("response", ""))
                    r["oracle_verdict"] = ev.get("verdict", "unknown")
                    r["oracle_method"] = "llm_eval"
                    r["oracle_reason"] = ev.get("reason", "")
                    n_llm += 1
            elif r.get("oracle_verdict") != "safe":
                r["oracle_verdict"] = "safe"
                r["oracle_method"] = "pre_blocked"
                r["oracle_reason"] = "defense_blocked"

        n_safe = sum(1 for r in details if r.get("oracle_verdict") == "safe")
        n = len(details)
        data["summary"] = {
            "dsr_block": round(sum(1 for r in details if r.get("final_action") == "block") / n * 100, 2),
            "dsr_oracle": round(n_safe / n * 100, 2),
            "n": n,
            "errors": sum(1 for r in details if r.get("final_action") == "error"),
        }
        data["oracle_model"] = ORACLE_MODEL
        json.dump(data, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

        # 同步 jsonl（权威断点）
        jsonl = p.with_suffix(p.suffix + ".jsonl")
        if jsonl.exists():
            with open(jsonl, "w", encoding="utf-8") as f:
                for r in details:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"{p.name}: n={n} allow={n_allow} (heuristic={n_heur}, llm={n_llm}) "
              f"| DSR(block)={data['summary']['dsr_block']}% DSR(oracle)={data['summary']['dsr_oracle']}%")


if __name__ == "__main__":
    main()
