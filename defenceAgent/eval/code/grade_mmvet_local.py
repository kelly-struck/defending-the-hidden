from __future__ import annotations
"""本地启发式评测器：在不依赖官方打分器的情况下，对 MM-Vet 预测结果进行本地打分，
并输出总体准确率与按 capability 的分项准确率。

用法示例（Windows PowerShell）：
  python -m defenceAgent.eval.code.grade_mmvet_local \
    --preds d:\\AgentScope\\defenceAgent\\eval\\results\\mmvet_baseline_preds.json \
    --dataset_dir d:\\AgentScope\\defenceAgent\\statics\\MM-Vet \
    --output d:\\AgentScope\\defenceAgent\\eval\\results\\mmvet_baseline_local_scored.json

preds 文件格式：list[{"id": str, "answer": str}]
将使用数据集中同 id 的 ground-truth（answer 与 capability）做启发式匹配。
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from defenceAgent.dataset_utils import load_mmvet_rows


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def _extract_numbers(s: str) -> List[float]:
    import re
    nums: List[float] = []
    for m in re.finditer(r"[-+]?\d*\.?\d+", s):
        try:
            nums.append(float(m.group(0)))
        except Exception:
            pass
    return nums


def _split_logic(gt: str) -> List[List[str]]:
    parts_and = [p.strip() for p in gt.split("<AND>")]
    groups: List[List[str]] = []
    for p in parts_and:
        opts = [o.strip() for o in p.split("<OR>")]
        groups.append([_normalize(o) for o in opts if o])
    return groups


def heuristic_correct(pred: str, gt: str) -> Tuple[bool, str]:
    p = _normalize(pred)
    g = _normalize(gt)

    if g in {"yes", "no"}:
        return (g in p), "yes/no match"

    if "<and>" in g or "<or>" in g:
        groups = _split_logic(gt)
        pnums = _extract_numbers(p)
        def _hit(opt: str) -> bool:
            if not opt:
                return False
            gnums = _extract_numbers(opt)
            if gnums and pnums:
                for gn in gnums:
                    for pn in pnums:
                        if abs(gn - pn) <= 0.5:
                            return True
            return opt in p
        ok = all(any(_hit(o) for o in ors) for ors in groups)
        return ok, "AND/OR heuristic"

    gnums = _extract_numbers(g)
    if gnums:
        pnums = _extract_numbers(p)
        for gn in gnums:
            for pn in pnums:
                if abs(gn - pn) <= 0.5:
                    return True, "numeric tolerance"
        return (any(str(int(gn)) in p for gn in gnums) or any(str(gn) in p for gn in gnums)), "numeric substring"

    if len(g.split()) <= 5:
        return (g in p), "substring"

    gtok = [t for t in g.split() if t.isalpha()][:3]
    hits = sum(1 for t in gtok if t in p)
    return (hits >= max(1, len(gtok)//2)), "token overlap"


def _ensure_cap_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def capability_breakdown(details: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    agg: Dict[str, Dict[str, int]] = {}
    for d in details:
        caps = _ensure_cap_list(d.get("capability")) or ["(uncategorized)"]
        ok = bool(d.get("correct"))
        for c in caps:
            st = agg.setdefault(c, {"total": 0, "correct": 0})
            st["total"] += 1
            if ok:
                st["correct"] += 1
    out: Dict[str, Dict[str, Any]] = {}
    for c, st in agg.items():
        total = st["total"]
        correct = st["correct"]
        out[c] = {"total": total, "correct": correct, "accuracy": (correct/total) if total else 0.0}
    return dict(sorted(out.items(), key=lambda kv: (-kv[1]["total"], kv[0])))


def main():
    ap = argparse.ArgumentParser(description="本地启发式 MM-Vet 打分（不依赖官方评测器）")
    ap.add_argument("--preds", required=True, help="预测文件：list[{id, answer}]")
    ap.add_argument("--dataset_dir", required=True, help="MM-Vet 数据集根目录（包含 data/test-*.parquet）")
    ap.add_argument("--output", default=None, help="保存评分报告到此路径（可选）")
    args = ap.parse_args()

    # 读取 GT 与 capability
    rows = load_mmvet_rows(args.dataset_dir, limit=None)
    idx: Dict[str, Dict[str, Any]] = {r["id"]: r for r in rows}

    # 读取预测
    preds: List[Dict[str, str]] = json.loads(Path(args.preds).read_text(encoding="utf-8"))

    total = correct = 0
    details: List[Dict[str, Any]] = []
    missing = 0

    for p in preds:
        rid = str(p.get("id"))
        ans = p.get("answer") or ""
        g = idx.get(rid)
        if not g:
            missing += 1
            continue
        gt = g.get("answer") or ""
        ok, reason = heuristic_correct(ans, gt)
        total += 1
        if ok:
            correct += 1
        details.append({
            "id": rid,
            "answer_pred": ans,
            "answer_gt": gt,
            "correct": bool(ok),
            "reason": reason,
            "capability": g.get("capability") or [],
        })

    report = {
        "mode": "local_grade",
        "total": total,
        "correct": correct,
        "accuracy": (correct/total) if total else 0.0,
        "not_found_in_dataset": missing,
        "details": details,
        "capability_breakdown": capability_breakdown(details),
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
