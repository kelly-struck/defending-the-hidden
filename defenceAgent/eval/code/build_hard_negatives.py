from __future__ import annotations
"""Build / update hard negative memory & rules from an evaluation report.

Usage:
  python -m defenceAgent.eval.code.build_hard_negatives --report_path report_mmsafetybench_30.json --out_memory memory/hard_negatives.jsonl --append

Logic:
  - Load JSON report (structure like mmsafetybench full-branches output).
  - Extract details entries where action == "allow" AND branch in target_risky_branches.
  - Simple heuristic classification into pattern buckets (finance / health / legal / gov / political / gambling / automation).
  - Emit JSONL lines with recommended_action (revise/block) and pattern_id.
    (We do not auto-modify YAML rules here; manual curation recommended for stability.)

Heuristics are intentionally conservative; future iterations can integrate reflection output.
"""
import argparse
import json
import re
from pathlib import Path

RISKY_BRANCHES = {
    "Financial_Advice": "financial_advice",
    "Health_Consultation": "unsafe_health",
    "Legal_Opinion": "legal_opinion",
    "Gov_Decision": "government_decision",
    "Political_Lobbying": "political_lobbying",
    "EconomicHarm": "economic_harm",
}

PATTERN_MAP = [
    ("finance_steps", "revise", ["portfolio", "invest", "allocation", "returns", "high accuracy", "predict", "buy", "sell", "strategy"]),
    ("health_dosage", "block", ["dosage", "mg", "treat", "prescription", "symptom", "diagnose", "side effects"]),
    ("legal_advice", "revise", ["sue", "liability", "contract", "jurisdiction", "evidence", "enforce"]),
    ("gov_decision", "revise", ["policy", "allocate", "surveillance", "public safety", "law enforcement"]),
    ("political_lobbying", "revise", ["campaign", "voters", "persuade", "donation", "lobby", "mobilize"]),
    ("gambling_prediction", "block", ["casino", "gambling", "betting", "horse race", "slot machine"]),
    ("ai_automation_sensitive", "block", ["automate", "without human", "predict", "model", "algorithm", "high accuracy"]),
]

def classify(text: str) -> tuple[str, str]:
    t = text.lower()
    for pid, act, kws in PATTERN_MAP:
        if any(kw in t for kw in kws):
            return pid, act
    return "generic_allow", "revise"  # default conservative

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_path", required=True)
    ap.add_argument("--out_memory", default="memory/hard_negatives.jsonl")
    ap.add_argument("--append", action="store_true", help="Append to existing JSONL if exists")
    args = ap.parse_args()

    report = json.loads(Path(args.report_path).read_text(encoding="utf-8"))
    lines = []
    for rep in report.get("reports", []):
        branch = rep.get("branch")
        if branch not in RISKY_BRANCHES:
            continue
        cat = RISKY_BRANCHES[branch]
        for det in rep.get("details", []):
            action = det.get("action")
            text = det.get("text")
            if action == "allow" and text:
                pid, rec_action = classify(text)
                entry = {
                    "text": text[:500],
                    "recommended_action": rec_action,
                    "pattern_id": pid,
                    "category": cat,
                    "rationale": "Auto-mined from allow sample (v1 heuristic)",
                }
                lines.append(entry)

    out_path = Path(args.out_memory)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for e in lines:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(json.dumps({"added": len(lines), "memory_path": str(out_path)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()