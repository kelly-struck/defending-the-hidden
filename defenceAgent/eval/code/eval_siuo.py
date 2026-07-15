from __future__ import annotations
"""Evaluate SIUO (Safe Inputs but Unsafe Output) with current defence pipeline and dump Q&A for manual judging.

- No automatic scoring. We only output the attacking question, model's final answer, and decision metadata for human evaluation.
- Supports two splits: gen (open-ended) and mcqa (multiple-choice).
- Uses DefenceOrchestrator for safety decisions and MLLMExecutor to produce final answers.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from agentscope.message import Msg, TextBlock, ImageBlock

from defenceAgent.dataset_utils import make_image_block_base64
from defenceAgent.orchestrator_debate import DebateOrchestrator
from defenceAgent.executor import MLLMExecutor


def load_siuo_rows(dataset_dir: str | Path, split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    if split == "gen":
        p = dataset_dir / "siuo_gen.json"
    elif split == "mcqa":
        p = dataset_dir / "siuo_mcqa.json"
    else:
        raise ValueError("split must be 'gen' or 'mcqa'")
    data = json.loads(p.read_text(encoding="utf-8"))
    if limit and limit > 0:
        data = data[:limit]
    return data


def build_user_msg(image_path: Path, question: str, options: Optional[Dict[str, str]] = None) -> Msg:
    # build blocks: prefer text then image for readability
    blocks: List[Dict[str, Any]] = []
    qtext = question.strip()
    if options:
        # Append options in a human-friendly format
        opts = "\n".join([f"{k}. {v}" for k, v in options.items()])
        qtext = f"{qtext}\n\nOptions:\n{opts}"
    blocks.append(TextBlock(type="text", text=qtext))
    img_bytes = image_path.read_bytes()
    blocks.append(ImageBlock(type="image", source={**make_image_block_base64(img_bytes)}))
    return Msg(name="user", role="user", content=blocks)


def normalize_action(meta: Any, fallback_safe: bool) -> str:
    action = None
    if isinstance(meta, dict):
        action = meta.get("action")
    if action is None:
        return "allow" if fallback_safe else "block"
    ra = str(action).lower()
    return ra if ra in ("allow", "revise") else "block"


def main():
    ap = argparse.ArgumentParser(description="Evaluate SIUO and dump Q&A for manual judging")
    ap.add_argument("--dataset_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "statics" / "SIUO"))
    ap.add_argument("--split", type=str, default="gen", choices=["gen", "mcqa"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    rows = load_siuo_rows(args.dataset_dir, args.split, None if args.limit is None or args.limit <= 0 else args.limit)

    orch = DebateOrchestrator()
    exe = MLLMExecutor()

    out: List[Dict[str, Any]] = []
    img_dir = Path(args.dataset_dir) / "images"

    for r in tqdm(rows, desc=f"SIUO:{args.split}"):
        qid = r.get("question_id")
        image_file = r.get("image")
        question = r.get("question") or ""
        category = r.get("category")
        options = r.get("options") if args.split == "mcqa" else None
        if not question.strip() or not image_file:
            continue
        img_path = img_dir / image_file
        if not img_path.is_file():
            tqdm.write(f"[skip id={qid}] image not found: {img_path}")
            continue

        user_msg = build_user_msg(img_path, question, options)

        # Run defence to get final decision
        try:
            debate_result = __import__("asyncio").run(orch.run(str(img_path), question))
        except Exception as e:
            tqdm.write(f"[skip id={qid}] {type(e).__name__}: {e}")
            continue
        
        # Extract decision and normalize action
        verdict = debate_result.get("verdict", "unsafe").lower()
        action = "allow" if verdict == "safe" else "block"
        meta = debate_result

        # Prepare answering path
        if action == "allow":
            user_msg_for_answer = user_msg
            judge_meta = {**meta, "action": "allow"}
        else:
            user_msg_for_answer = user_msg
            judge_meta = {**meta, "action": "block", "user_notice": "抱歉，该请求不符合使用政策。"}

        # Get final answer (block → refusal text)
        try:
            answer_msg = __import__("asyncio").run(exe(user_msg_for_answer, judge_meta))
            answer_text = answer_msg.content if isinstance(answer_msg.content, str) else str(answer_msg.content)
            # 如果影子闸门触发并标记为 revise，则将最终动作重标为 revise 计入防御成功
            if isinstance(answer_msg.metadata, dict) and answer_msg.metadata.get("shadow_gate") and answer_msg.metadata.get("action") == "revise":
                action = "revise"
        except Exception as e:
            answer_text = f"[executor_error] {type(e).__name__}: {e}"

        rec: Dict[str, Any] = {
            "id": qid,
            "image": image_file,
            "image_path": str(img_path),
            "category": category,
            "question": question,
            "final_action": action,
            "answer": answer_text,
        }
        if options:
            rec["options"] = options
        if isinstance(meta, dict):
            rec["decision_meta"] = meta
            if meta.get("revised_text") and action == "revise":
                rec["question_revised"] = meta.get("revised_text")
        out.append(rec)

    report = {
        "dataset": "SIUO",
        "split": args.split,
        "total": len(out),
        "details": out,
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"Wrote {len(out)} records to {args.output}")


if __name__ == "__main__":
    main()
