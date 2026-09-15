# -*- coding: utf-8 -*-
"""Phase 1 (E7) C3: 逐样本 oracle–human Cohen's κ。

- oracle 逐样本判定来自 run_siuo_with_records.py --oracle 的输出
  （results/siuo_full_system_oracle.json 或 C1 的 run 输出）
- 人工标注来自 results/human_validation/annotations.json：
    [{"id": <question_id>, "annotator": 1|2|3, "human_safe": true|false}, ...]
- 输出：每位标注者的 oracle–human Cohen's κ（逐样本），附 Fleiss' κ 对照
  （Fleiss' κ 若 sklearn 无直接实现则用 run_human_validation 的既有逻辑）。
"""
from __future__ import annotations
import os
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root

from sklearn.metrics import cohen_kappa_score


def fleiss_kappa(ratings: list[list[int]]) -> float:
    """Fleiss' κ：ratings[i][j] = 第 i 样本被判为类别 j 的人数（2 类）。"""
    import math
    n_samples = len(ratings)
    n_raters = sum(ratings[0]) if ratings else 0
    n_cat = len(ratings[0]) if ratings else 2
    if n_samples == 0 or n_raters <= 1 or n_raters == n_cat:
        return float("nan")
    p_j = [sum(r[j] for r in ratings) / (n_samples * n_raters) for j in range(n_cat)]
    P_i = []
    for r in ratings:
        s = sum(r)
        P_i.append((sum(x * x for x in r) - s) / (s * (s - 1)) if s > 1 else 1.0)
    P_bar = sum(P_i) / n_samples
    P_e = sum(p * p for p in p_j)
    if P_e == 1.0:
        return float("nan")
    return (P_bar - P_e) / (1 - P_e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, help="full_system --oracle 输出 JSON（含 details）")
    ap.add_argument("--annotations", required=True, help="人工标注 JSON")
    ap.add_argument("--output", default=r"d:\AgentScope\results\oracle_human_kappa.json")
    a = ap.parse_args()

    oracle_data = json.load(open(a.oracle, encoding="utf-8"))
    details = oracle_data.get("details", oracle_data) if isinstance(oracle_data, dict) else oracle_data
    oracle = {str(r.get("id")): (r.get("oracle_verdict") == "safe") for r in details}

    ann_data = json.load(open(a.annotations, encoding="utf-8"))
    if isinstance(ann_data, dict):
        ann_data = ann_data.get("annotations", ann_data.get("details", []))

    ann_by_id = {}
    for a_ in ann_data:
        ann_by_id.setdefault(str(a_.get("id")), []).append(a_)

    annotators = sorted(set(str(a_.get("annotator")) for a_ in ann_data))
    out = {"oracle_total": len(oracle), "annotated_total": len(ann_data), "per_annotator": {}}

    print(f"oracle 样本数: {len(oracle)} | 标注条数: {len(ann_data)} | 标注者: {annotators}")
    for anno_id in annotators:
        rows = [a_ for a_ in ann_data if str(a_.get("annotator")) == anno_id]
        pairs = []
        missing = 0
        for a_ in rows:
            sid = str(a_.get("id"))
            if sid not in oracle:
                missing += 1
                continue
            pairs.append((oracle[sid], bool(a_.get("human_safe"))))
        if not pairs:
            print(f"  Annotator {anno_id}: 无配对样本（missing oracle id={missing}）")
            continue
        y_oracle = [p[0] for p in pairs]
        y_human = [p[1] for p in pairs]
        k = cohen_kappa_score(y_oracle, y_human)
        agree = sum(1 for o, h in pairs if o == h) / len(pairs)
        out["per_annotator"][anno_id] = {
            "n": len(pairs),
            "oracle_human_kappa": round(k, 4),
            "agreement": round(agree, 4),
            "missing": missing,
        }
        print(f"  Annotator {anno_id}: oracle–human Cohen's κ = {k:.3f} (n={len(pairs)}, agreement={agree:.1%})")

    # Fleiss' κ 对照（3 标注者、167 样本、2 类）
    ids = sorted({str(a_.get("id")) for a_ in ann_data} & set(oracle.keys()))
    n_ann = len(annotators)
    if n_ann >= 2 and ids:
        ratings = []
        for sid in ids:
            votes = [a_ for a_ in ann_data if str(a_.get("id")) == sid]
            safe_cnt = sum(1 for v in votes if v.get("human_safe"))
            ratings.append([n_ann - safe_cnt, safe_cnt])
        fk = fleiss_kappa(ratings)
        out["fleiss_kappa_human"] = round(fk, 4) if fk == fk else None
        print(f"  Fleiss' κ (标注者间, {n_ann} 标注者, n={len(ids)}) = {fk:.3f}")

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"-> {a.output}")


if __name__ == "__main__":
    main()
