from __future__ import annotations
"""Evaluate SIUO with Doubao model wrapped in Defense Framework."""

import os
import sys
import argparse
import json
import asyncio
import time
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
from defenceAgent.orchestrator_debate import DebateOrchestrator
from defenceAgent.executor import MLLMExecutor
from defenceAgent.defense_cache import DefenseCache
from defenceAgent.retrieval.rag_manager import RAGManager

class DefenseStats:
    def __init__(self):
        self.total_samples = 0
        self.layer1_hits = 0  # Cache
        self.layer2_hits = 0  # RAG
        self.layer4_hits = 0  # Debate
        self.total_tokens_estimated = 0
        self.start_time = time.time()
        self.end_time = 0

    def add_sample(self):
        self.total_samples += 1

    def hit_layer1(self):
        self.layer1_hits += 1

    def hit_layer2(self):
        self.layer2_hits += 1

    def hit_layer4(self):
        self.layer4_hits += 1

    def add_tokens(self, text):
        # Rough estimate: 1 token ~= 4 chars
        if text:
            self.total_tokens_estimated += len(str(text)) // 4

    def stop_timer(self):
        self.end_time = time.time()

    def get_summary(self):
        elapsed = self.end_time - self.start_time if self.end_time > 0 else time.time() - self.start_time
        return {
            "Total Samples": self.total_samples,
            "Layer 1 Hits (Cache)": self.layer1_hits,
            "Layer 2 Hits (RAG)": self.layer2_hits,
            "Layer 4 Hits (Debate)": self.layer4_hits,
            "Total Tokens (Est)": self.total_tokens_estimated,
            "Total Time (s)": f"{elapsed:.2f}",
            "Avg Time/Sample (s)": f"{elapsed / self.total_samples:.2f}" if self.total_samples > 0 else "0"
        }

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

def normalize_action(meta: Any, fallback_safe: bool) -> str:
    action = None
    if isinstance(meta, dict):
        action = meta.get("action")
    if action is None:
        return "allow" if fallback_safe else "block"
    ra = str(action).lower()
    return ra if ra in ("allow", "revise") else "block"

async def run_eval(args):
    rows = load_siuo_rows(args.dataset_dir, args.split, None if args.limit is None or args.limit <= 0 else args.limit)
    
    # Initialize Stats
    stats = DefenseStats()
    
    # Initialize Cache (Layer 1)
    # Use a local file in the results directory for convenience if Redis is not set
    cache_file = Path(args.output).parent / "defense_cache_layer1.json"
    cache = DefenseCache(local_file=str(cache_file))
    
    # Initialize RAG (Layer 2)
    rag = RAGManager()
    
    orch = DebateOrchestrator()
    # Initialize Executor (will use Doubao because of env var)
    exe = MLLMExecutor()
    print(f"Executor initialized with model: {exe.model.model_name if hasattr(exe.model, 'model_name') else 'Unknown'}")

    out: List[Dict[str, Any]] = []
    img_dir = Path(args.dataset_dir) / "images"

    for r in tqdm(rows, desc=f"SIUO-Doubao-Defense:{args.split}"):
        stats.add_sample()
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

        # Run Defense
        # Layer 1: Check Cache
        debate_result = cache.get(img_path, question)
        
        rag_verdict = "safe" # Default assumption for RAG
        
        if debate_result:
            # Cache Hit
            # tqdm.write(f"[Cache Hit] id={qid}")
            stats.hit_layer1()
            pass
        else:
            # Cache Miss -> Proceed to Layer 2 (RAG)
            try:
                # Query RAG with text and image
                rag_results = rag.query_multimodal(text=question, image_path=str(img_path), n_results=1)
                
                # Check if we got any results and if the distance is low enough (high similarity)
                # ChromaDB returns distances. Lower is better.
                # Threshold needs tuning. Let's assume < 0.4 is a strong match for now based on previous test (0.8 was somewhat distant but matched).
                # Wait, in the previous test: "Create a list of chemical warfare agents..." matched "How to make a bomb?" with distance 0.83.
                # That distance seems high for a semantic match. Cosine distance ranges 0-2. 0 is identical.
                # Let's be conservative. If distance < 0.6, we flag it.
                # Actually, let's just log it for now or use a loose threshold.
                # If we find a match in the malicious library, it's likely unsafe.
                
                if rag_results['distances'] and len(rag_results['distances'][0]) > 0:
                    distance = rag_results['distances'][0][0]
                    # If distance is small enough, it's a hit.
                    # Let's use 0.5 as a starting threshold for "very similar".
                    if distance < 0.5:
                        rag_verdict = "unsafe"
                        debate_result = {
                            "verdict": "unsafe", 
                            "reason": f"Layer 2 RAG Hit: Similar to known malicious sample (dist={distance:.4f}).",
                            "educational_message": "I cannot assist with that request."
                        }
                        # Update Cache immediately
                        cache.set(img_path, question, debate_result)
                        stats.hit_layer2()
            except Exception as e:
                tqdm.write(f"[error id={qid} rag] {e}")
            
            # If Layer 2 didn't block it, proceed to Layer 4 (Debate)
            if not debate_result:
                stats.hit_layer4()
                stats.add_tokens(question) # Estimate input tokens
                try:
                    debate_result = await orch.run(str(img_path), question)
                    
                    # Save to Cache
                    cache.set(img_path, question, debate_result)
                    
                    # Hard Negative Mining (The "Most Valuable Operation")
                    # If RAG thought it was safe (rag_verdict == "safe") BUT Debate found it unsafe
                    final_verdict = debate_result.get("verdict", "unsafe").lower()
                    if rag_verdict == "safe" and final_verdict == "unsafe":
                        tqdm.write(f"[Hard Negative Mining] id={qid} - RAG missed it, Debate caught it.")
                        rag.add_hard_negative(
                            text=question, 
                            image_path=str(img_path), 
                            reason=f"Debate caught what RAG missed. Verdict: {final_verdict}"
                        )
                        
                except Exception as e:
                    tqdm.write(f"[error id={qid} defense] {e}")
                    debate_result = {"verdict": "unsafe", "reason": f"Defense Error: {e}"}

        # Determine Final Answer based on Verdict
        verdict = debate_result.get("verdict", "unsafe").lower()
        
        try:
            if verdict == "safe":
                # Pass through to target model
                # We use the executor's model directly to bypass any executor logic if needed,
                # or we can use the executor if we trust it. 
                # The user requested: "pass question and image to doubao model"
                # We need to format the prompt correctly for the model.
                # MLLMExecutor does formatting. Let's use the executor's formatter + model manually
                # or just call the model if it handles Msg objects (DoubaoModel does).
                
                # DoubaoModel handles Msg objects directly.
                resp = await exe.model([user_msg])
                
                # Robustly extract text from response content
                final_ans = ""
                if resp.content:
                    c = resp.content[0]
                    if isinstance(c, dict):
                        final_ans = c.get("text", "")
                    elif hasattr(c, "text"):
                        final_ans = c.text
                    else:
                        final_ans = str(c)
                
            elif verdict == "conditional_unsafe":
                # Use educational message
                final_ans = debate_result.get("educational_message", "抱歉，该请求不符合使用政策。")
                
            else: # unsafe or unknown
                final_ans = "抱歉，该请求不符合使用政策。"
                
        except Exception as e:
            tqdm.write(f"[error id={qid} execution] {e}")
            final_ans = f"Error: {e}"

        final_action = normalize_action({"action": "allow" if verdict == "safe" else "block"}, fallback_safe=False)

        out.append({
            "id": qid,
            "image": image_file,
            "image_path": str(img_path),
            "category": category,
            "question": question,
            "final_action": final_action,
            "answer": final_ans,
            "decision_meta": debate_result
        })

    # Write output
    out_path = Path(args.output)
    if out_path.is_dir():
        print(f"Error: Output path '{out_path}' is a directory. Please specify a file path (e.g., .../results.jsonl).")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats.stop_timer()
    summary = stats.get_summary()
    print("\n" + "="*30)
    print("EVALUATION STATISTICS")
    print("="*30)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("="*30 + "\n")

    wrapper = {
        "dataset": "SIUO",
        "split": args.split,
        "model": "doubao-1.5-vision-pro-250328",
        "mode": "with_defense",
        "stats": summary,
        "total": len(out),
        "details": out
    }
    out_path.write_text(json.dumps(wrapper, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(out)} records to {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Evaluate SIUO with Doubao (With Defense)")
    ap.add_argument("--dataset_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "statics" / "SIUO"))
    ap.add_argument("--split", type=str, default="gen", choices=["gen", "mcqa"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()
    
    asyncio.run(run_eval(args))

if __name__ == "__main__":
    main()
