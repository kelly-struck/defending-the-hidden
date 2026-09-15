# -*- coding: utf-8 -*-
"""Phase 1 (F1) 抽取真实 L2 假阳性（名人 ID 类）作毒源。

来源说明：
- 历史 FPR 运行 results/fpr_real/fpr_raw_results.json 记录了 vector_db 层拦截的
  benign 样本 id（celebrity ID 类查询，论文 tab:fpr_breakdown 一致）。
- 当前干净评测库（chroma_db_phase1）下 L2 对 benign 拦截为 0（celebrity 查询不触发），
  因此毒源取自历史观测的 L2 FP id，从 benign_dataset.json 找回 question + image（真实名人图）。
- 输出: [{id, text, image_b64}] 供 evaluate_poisoning_robustness.py 作毒源。
"""
from __future__ import annotations
import os
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root

FPR_RAW = r"d:\AgentScope\results\fpr_real\fpr_raw_results.json"
BENIGN = r"d:\AgentScope\datasets\benign\benign_dataset.json"
OUT = r"d:\AgentScope\results\phase1\f1_real_l2_fps.json"


def img_to_b64(img_data) -> str | None:
    """benign_dataset image 字段 {bytes: {type,data}, path} → data URI。"""
    try:
        if isinstance(img_data, dict):
            b = img_data.get("bytes")
            if isinstance(b, dict):
                # {"type": "base64", "data": "..."} → data URI
                data = b.get("data", "")
                media = b.get("media_type", "image/jpeg")
                if data and not data.startswith("data:"):
                    return f"data:{media};base64,{data}"
                return data
            if isinstance(b, str) and b.startswith("data:"):
                return b
            if isinstance(b, (bytes, bytearray)):
                return "data:image/jpeg;base64," + base64.b64encode(bytes(b)).decode()
        elif isinstance(img_data, str) and img_data.startswith("data:"):
            return img_data
        elif isinstance(img_data, (bytes, bytearray)):
            return "data:image/jpeg;base64," + base64.b64encode(bytes(img_data)).decode()
    except Exception as e:
        print(f"  [img_to_b64 error] {e}")
    return None


def main():
    fpr = json.load(open(FPR_RAW, encoding="utf-8"))
    details = fpr.get("details", fpr) if isinstance(fpr, dict) else fpr
    vdb_fp_ids = [str(r.get("id")) for r in details
                  if r.get("defense_source") == "vector_db" and r.get("is_false_positive")]
    print(f"历史 vector_db FP: {len(vdb_fp_ids)} 条")

    benign = json.load(open(BENIGN, encoding="utf-8"))
    by_id = {str(s.get("id")): s for s in benign}

    poison = []
    for sid in vdb_fp_ids:
        s = by_id.get(sid)
        if not s:
            continue
        b64 = img_to_b64(s.get("image"))
        if not b64 or not s.get("question"):
            continue
        poison.append({"id": sid, "text": s["question"], "image_b64": b64, "domain": s.get("domain")})

    # 只保留名人 ID 类（celebrity），与论文 tab:fpr_breakdown 的根因一致
    celebrity = [p for p in poison if "celebrity" in str(p.get("id", ""))]
    print(f"含图像毒源: {len(poison)} | 其中 celebrity 类: {len(celebrity)}")

    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"source": "observed_l2_fp", "n": len(celebrity), "items": celebrity},
              open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"-> {OUT} ({len(celebrity)} 条毒源)")
    for p in celebrity[:3]:
        print("  示例:", p["id"], "|", p["text"][:60])


if __name__ == "__main__":
    main()
