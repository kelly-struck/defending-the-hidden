from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from agentscope.message import Msg, ImageBlock, TextBlock, Base64Source, URLSource

from .orchestrator import DefenceOrchestrator
from .main import load_image_source


def build_user_msg(image: Optional[str], text: Optional[str]) -> Msg:
    blocks = []
    if text and len(text.strip()) > 0:
        blocks.append(TextBlock(type="text", text=text))
    if image and len(image.strip()) > 0:
        src = load_image_source(image)
        blocks.append(ImageBlock(type="image", source=src))
    if not blocks:
        raise ValueError("CSV 行缺少 image 与 text，至少提供其一")
    return Msg(name="user", role="user", content=blocks)


def decide_action(final_safe: bool, judge_meta: dict | None) -> str:
    if not final_safe:
        return "block"
    if isinstance(judge_meta, dict):
        return str(judge_meta.get("action", "allow"))
    return "allow"


def run_cases(cases: List[Dict[str, str]]) -> Dict[str, object]:
    orch = DefenceOrchestrator()
    import asyncio

    results = []
    passed = 0
    skipped = 0

    for i, row in enumerate(cases, 1):
        image = row.get("image") or None
        text = row.get("text") or None
        expect = (row.get("expect") or "").strip().lower() or None

        # 跳过无内容的行
        if (not image or len(image.strip()) == 0) and (not text or len(text.strip()) == 0):
            skipped += 1
            continue

        user_msg = build_user_msg(image, text)
        final, img_res, txt_res = asyncio.run(orch.run(user_msg))
        pred_action = decide_action(final.safe, final.raw.metadata if final.raw else None)
        ok = (expect is None) or (expect == pred_action) or (
            expect in {"block", "deny"} and pred_action == "block"
        )
        passed += 1 if ok else 0

        results.append(
            {
                "index": i,
                "image": image,
                "text": text,
                "expect": expect,
                "pred_action": pred_action,
                "final_safe": final.safe,
                "judge_meta": final.raw.metadata if final.raw else None,
            }
        )

    total = len(cases) - skipped
    acc = passed / total if total else 0.0
    return {
        "total": total,
        "passed": passed,
        "skipped": skipped,
        "accuracy": acc,
        "details": results,
    }


def read_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v or "").strip() for k, v in r.items()})
    return rows


def main():
    parser = argparse.ArgumentParser(description="DefenceAgent batch runner")
    parser.add_argument("-i", "--input", required=True, help="CSV 文件，列: image,text,expect")
    parser.add_argument("-o", "--output", default=None, help="输出 JSON 报告路径")
    args = parser.parse_args()

    cases = read_csv(args.input)
    report = run_cases(cases)

    text_report = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text_report, encoding="utf-8")
    print(text_report)


if __name__ == "__main__":
    main()
