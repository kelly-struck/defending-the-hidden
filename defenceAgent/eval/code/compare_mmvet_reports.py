from __future__ import annotations
"""Combine baseline and defence MM-Vet reports into a single comparison JSON.

Usage:
  python -m defenceAgent.eval.code.compare_mmvet_reports \
    --baseline d:\\AgentScope\\defenceAgent\\eval\\results\\mmvet_baseline_official.json \
    --defence  d:\\AgentScope\\defenceAgent\\eval\\results\\mmvet_defence_official.json \
    --output   d:\\AgentScope\\defenceAgent\\eval\\results\\mmvet_compare.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _index_details_by_id(details: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for d in details or []:
        rid = str(d.get("id", ""))
        if rid:
            out[rid] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--defence", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    defence = json.loads(Path(args.defence).read_text(encoding="utf-8"))

    cmp_obj: Dict[str, Any] = {
        "summary": {
            "baseline": {
                "total": baseline.get("total"),
                "correct": baseline.get("correct"),
                "accuracy": baseline.get("accuracy"),
            },
            "defence": {
                "total": defence.get("total"),
                "correct": defence.get("correct"),
                "accuracy": defence.get("accuracy"),
                "allowed": defence.get("allowed"),
                "blocked": defence.get("blocked"),
                "revised": defence.get("revised"),
            },
        },
        "official": {
            "baseline": baseline.get("official_evaluator"),
            "defence": defence.get("official_evaluator"),
        },
    }

    # Optional: per-id diffs
    b_idx = _index_details_by_id(baseline.get("details") or [])
    d_idx = _index_details_by_id(defence.get("details") or [])
    ids = sorted(set(b_idx.keys()) | set(d_idx.keys()))
    diffs: List[Dict[str, Any]] = []
    for rid in ids:
        b = b_idx.get(rid)
        d = d_idx.get(rid)
        if not (b and d):
            continue
        diffs.append({
            "id": rid,
            "question": d.get("question") if d else (b.get("question") if b else None),
            "gt": d.get("answer_gt") if d else (b.get("answer_gt") if b else None),
            "baseline": {
                "answer": b.get("answer_pred"),
                "correct": b.get("correct"),
            },
            "defence": {
                "answer": d.get("answer_pred"),
                "correct": d.get("correct"),
                "action": d.get("action"),
            },
        })

    cmp_obj["diffs"] = diffs

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(cmp_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(cmp_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
