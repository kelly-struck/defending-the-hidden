from __future__ import annotations
"""Evaluate SIUO with Doubao model directly (No Defense)."""

import os
import sys
import argparse
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

# Set target model to Doubao BEFORE importing defenceAgent
os.environ["DEF_TARGET_MODEL"] = "doubao-1.5-vision-pro-250328"

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from agentscope.message import Msg, TextBlock, ImageBlock
from defenceAgent.dataset_utils import make_image_block_base64
# We only need MLLMExecutor for no-defense run
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
    blocks: List[Dict[str, Any]] = []
    qtext = question.strip()
    if options:
        opts = "\n".join([f"{k}. {v}" for k, v in options.items()])
        qtext = f"{qtext}\n\nOptions:\n{opts}"
    blocks.append(TextBlock(type="text", text=qtext))
    img_bytes = image_path.read_bytes()
    blocks.append(ImageBlock(type="image", source={**make_image_block_base64(img_bytes)}))
    return Msg(name="user", role="user", content=blocks)

async def run_eval(args):
    rows = load_siuo_rows(args.dataset_dir, args.split, None if args.limit is None or args.limit <= 0 else args.limit)
    
    # Initialize Executor (will use Doubao because of env var)
    exe = MLLMExecutor()
    print(f"Executor initialized with model: {exe.model.model_name if hasattr(exe.model, 'model_name') else 'Unknown'}")

    out: List[Dict[str, Any]] = []
    img_dir = Path(args.dataset_dir) / "images"

    for r in tqdm(rows, desc=f"SIUO-Doubao-NoDefense:{args.split}"):
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

        # NO DEFENSE: Always allow
        # We pass an empty judge_meta with action="allow"
        fake_judge_meta = {"action": "allow"}
        
        try:
            final_res = await exe(user_msg, fake_judge_meta)
            final_ans = final_res.content
        except Exception as e:
            tqdm.write(f"[error id={qid}] {e}")
            final_ans = f"Error: {e}"

        out.append({
            "id": qid,
            "image": image_file,
            "image_path": str(img_path),
            "category": category,
            "question": question,
            "final_action": "allow", # Always allow in no-defense
            "answer": final_ans,
            "decision_meta": fake_judge_meta
        })

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = {
        "dataset": "SIUO",
        "split": args.split,
        "model": "doubao-1.5-vision-pro-250328",
        "mode": "no_defense",
        "total": len(out),
        "details": out
    }
    out_path.write_text(json.dumps(wrapper, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(out)} records to {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Evaluate SIUO with Doubao (No Defense)")
    ap.add_argument("--dataset_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "statics" / "SIUO"))
    ap.add_argument("--split", type=str, default="gen", choices=["gen", "mcqa"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()
    
    asyncio.run(run_eval(args))

if __name__ == "__main__":
    main()
