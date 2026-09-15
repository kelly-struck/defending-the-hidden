from __future__ import annotations
import argparse, json, statistics
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    details = data.get("details", [])

    total = len(details)
    c_rev = sum(1 for r in details if r.get("final_action") == "revise")
    c_blk = sum(1 for r in details if r.get("final_action") == "block")
    c_all = sum(1 for r in details if r.get("final_action") == "allow")
    success = c_rev + c_blk
    success_rate = (success / total) if total else 0.0

    fast_path = sum(1 for r in details if isinstance(r.get("decision_meta"), dict) and r["decision_meta"].get("source") == "fast_path")
    perfs = [((r.get("decision_meta", {}).get("perf", {}) or {}).get("total_ms")) for r in details]
    perfs = [x for x in perfs if isinstance(x, (int, float))]
    avg_total_ms = (sum(perfs) / len(perfs)) if perfs else 0.0
    median_total_ms = statistics.median(perfs) if perfs else 0.0
    judge_skipped = sum(1 for r in details if (r.get("decision_meta", {}).get("perf", {}) or {}).get("judge_skipped"))

    by_category = defaultdict(lambda: Counter())
    for r in details:
        cat = r.get("category") or "unknown"
        by_category[cat][r.get("final_action")] += 1

    out = {
        "file": args.input,
        "total": total,
        "revise": c_rev,
        "block": c_blk,
        "allow": c_all,
        "success": success,
        "success_rate": success_rate,
        "fast_path": fast_path,
        "avg_total_ms": avg_total_ms,
        "median_total_ms": median_total_ms,
        "judge_skipped": judge_skipped,
        "by_category": {k: dict(v) for k, v in by_category.items()},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
