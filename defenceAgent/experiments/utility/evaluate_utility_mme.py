sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root

import asyncio
import pandas as pd
import os
import random
import re
from typing import List, Tuple
from agentscope.message import Msg, ImageBlock, TextBlock, Base64Source
import io
import base64
import json
from PIL import Image

# Import existing modules
from defenceAgent.models import ModelRegistry
from defenceAgent.orchestrator import DefenceOrchestrator
from defenceAgent.executor import MLLMExecutor
from defenceAgent.retrieval.rag_manager import RAGManager # Added for reset logic

# Configuration
PARQUET_DIR = r"d:\AgentScope\defenceAgent\statics\MME\data"
TARGET_MODELS = ["qwen-vl-plus", "doubao-1.5-vision-pro-250328"]
SAMPLE_PER_CATEGORY_PER_FILE = 5  # Estimated ~10 files * 5 approx 50, need logic to sample efficiently
TOTAL_SAMPLES_TARGET = 300 

RESULT_DIR = "results"

def load_data(total_samples=300):
    all_files = [os.path.join(PARQUET_DIR, f) for f in os.listdir(PARQUET_DIR) if f.endswith(".parquet")]
    
    # We want a diverse set, so let's read all into a list (metadata only if possible, but images are embedded)
    # Since reading all parsers is heavy, let's stream-read or sample per file.
    
    collected_rows = []
    
    print(f"Loading data from {len(all_files)} parquet files...")
    
    # Simple strategy: Load each file (they are split), sample equal amount to reach total
    samples_per_file = max(1, total_samples // len(all_files))
    
    columns_needed = ['question_id', 'question', 'answer', 'image', 'category']
    
    for pf in all_files:
        try:
            df = pd.read_parquet(pf, columns=columns_needed)
            # Remove rows where answer is not strict Yes/No if possible? 
            # MME answers are usually "Yes" or "No".
            
            # Sample
            if len(df) > samples_per_file:
                sampled = df.sample(n=samples_per_file, random_state=42)
            else:
                sampled = df
            
            collected_rows.append(sampled)
        except Exception as e:
            print(f"Error reading {pf}: {e}")
            
    if not collected_rows:
        raise ValueError("No data loaded!")
        
    full_df = pd.concat(collected_rows, ignore_index=True)
    
    # Trim to exact target if exceeded
    if len(full_df) > total_samples:
        full_df = full_df.sample(n=total_samples, random_state=42)
        
    print(f"Total samples prepared: {len(full_df)}")
    return full_df

def heuristic_parse_yes_no(text: str) -> str:
    """Extract Yes/No from model response."""
    if not text:
        return "Unknown"
    
    # Normalize
    text = text.lower().strip()
    
    # 1. Clean punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # 2. Check strict start
    if text.startswith('yes'):
        return "Yes"
    if text.startswith('no'):
        return "No"
        
    # 3. Check loose contains (riskier but needed for verbose models)
    # "The answer is yes." or "I think it is no."
    tokens = text.split()
    if 'yes' in tokens:
        return "Yes"
    if 'no' in tokens:
        return "No"
        
    return "Other"

def extract_base64_from_row(row):
    img_data = row['image']
    if isinstance(img_data, dict) and 'bytes' in img_data:
        b_data = img_data['bytes']
    elif isinstance(img_data, bytes):
        b_data = img_data
    else:
        # Fallback for some hf datasets
        return None
        
    return base64.b64encode(b_data).decode('utf-8')

def compress_image(b64_str, max_size_mb=4):
    """Compress image if larger than max_size_mb."""
    if not b64_str:
        return b64_str
    
    try:
        # Check size roughly (base64 is 4/3 larger than binary)
        if len(b64_str) < max_size_mb * 1024 * 1024 * 1.35:
            return b64_str

        img_data = base64.b64decode(b64_str)
        if len(img_data) <= max_size_mb * 1024 * 1024:
            return b64_str
            
        # Compress
        img = Image.open(io.BytesIO(img_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # 1. Try resize first if huge
        width, height = img.size
        # Aggressive initial resize if very large
        if width > 2048 or height > 2048:
            img.thumbnail((2048, 2048))
            
        # 2. JPEG compression loop
        for quality in [85, 70, 50]:
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality)
            compressed_data = output.getvalue()
            if len(compressed_data) <= max_size_mb * 1024 * 1024:
                return base64.b64encode(compressed_data).decode('utf-8')
        
        # 3. Aggressive resize if still failing
        output = io.BytesIO()
        img.thumbnail((1024, 1024))
        img.save(output, format='JPEG', quality=40)
        return base64.b64encode(output.getvalue()).decode('utf-8')
        
    except Exception as e:
        print(f"Image compression warning: {e}")
        return b64_str

async def run_single_row(row, executor_baseline, orchestrator_defense, run_mode='defense'):
    """
    run_mode: 'baseline' or 'defense'
    """
    question = row['question']
    b64_img = extract_base64_from_row(row)
    
    if b64_img:
        b64_img = compress_image(b64_img, max_size_mb=4)

    blocks = [TextBlock(type="text", text=question)]
    if b64_img:
        blocks.append(ImageBlock(type="image", source=Base64Source(type="base64", media_type="image/jpeg", data=b64_img)))
    
    user_msg = Msg(name="user", role="user", content=blocks)
    
    try:
        if run_mode == 'baseline':
            resp = await executor_baseline(user_msg, judge_meta=None)
            return resp.content, "Allowed"
            
        elif run_mode == 'defense':
            final_judge_msg, img_safety, txt_safety = await orchestrator_defense.run(user_msg)
            
            judge_meta = final_judge_msg.raw.metadata if (final_judge_msg and final_judge_msg.raw) else {}
            action = judge_meta.get("action", "allow")
            
            if action == 'block':
                return "REFUSAL_BY_DEFENSE", "Blocked"
            
            final_resp = await executor_baseline(user_msg, judge_meta)
            return final_resp.content, action
            
    except Exception as e:
        error_str = str(e)
        if "SSLError" in error_str or "Max retries exceeded" in error_str or "RemoteDisconnected" in error_str or "ProxyError" in error_str or "ConnectionPool" in error_str:
             print(f"  [Network Error Ignored] treating as Allowed: {e}")
             return "Network Error (Ignored)", "Allowed"
        
        if "BadRequest.TooLarge" in error_str:
            print(f"  [Error] Image too large even after compression.")
        
        print(f"Error processing row: {e}")
        return f"ERROR: {e}", "Error"

async def main():
    # --- ENVIRONMENT RESET START ---
    # Configure a fresh RAG collection for MME to avoid pollution
    exp_collection = "mme_utility_clean_v1"
    os.environ["RAG_COLLECTION_NAME"] = exp_collection
    print(f"\n[Validation Setup] Using isolated RAG collection: {exp_collection}")
    
    # Force clear the RAG collection (Layer 2)
    try:
        tmp_rag = RAGManager(collection_name=exp_collection)
        try:
            tmp_rag.client.delete_collection(exp_collection)
            print(f"[Validation Setup] Successfully deleted existing RAG collection: {exp_collection}")
        except Exception:
            pass
    except Exception as e:
        print(f"[Validation Setup] Warning during RAG reset: {e}")
    # --- ENVIRONMENT RESET END ---

    # 1. Prepare Data
    df = load_data(TOTAL_SAMPLES_TARGET)
    
    # 2. Setup Results Storage (Not used globally anymore)
    
    # 3. Iterate Models
    for model_name in TARGET_MODELS:
        print(f"\nEvaluating Model: {model_name}")
        model_results = [] # Store results for current model only
        
        executor = MLLMExecutor(target_model_name=model_name)
        orchestrator = DefenceOrchestrator() # Uses default config for defense agents
        
        # --- CACHE RESET (Layer 1) ---
        if orchestrator.cache:
            orchestrator.cache.local_cache = {}
            if getattr(orchestrator.cache, 'use_redis', False) and list(orchestrator.cache.redis_client.keys()):
                 try:
                     orchestrator.cache.redis_client.flushdb()
                 except Exception as e:
                     print(f"Redis flush failed: {e}")
            print(f"[Validation Setup] Layer 1 (Semantic Cache) cleared for {model_name}.")
        # -----------------------------
        
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"Processing {idx}/{len(df)}...")
                
            row_res = {
                "question_id": row['question_id'],
                "category": row['category'],
                "ground_truth": row['answer'],
                "model": model_name
            }
            
            # A. Baseline Run
            base_ans_text, base_action = await run_single_row(row, executor, orchestrator, run_mode='baseline')
            row_res["baseline_raw_response"] = base_ans_text
            
            # Grading Baseline
            if "Network Error (Ignored)" in base_ans_text:
                row_res["baseline_parsed_answer"] = "NetworkError"
                row_res["baseline_is_correct"] = True
            else:
                row_res["baseline_parsed_answer"] = heuristic_parse_yes_no(base_ans_text)
                row_res["baseline_is_correct"] = (row_res["baseline_parsed_answer"] == row['answer'])
            
            # B. Defense Run
            def_ans_text, def_action = await run_single_row(row, executor, orchestrator, run_mode='defense')
            row_res["defense_raw_response"] = def_ans_text
            row_res["defense_action"] = def_action 
            
            # Grading Defense
            if def_action == "Blocked":
                 row_res["defense_parsed_answer"] = "Blocked"
                 row_res["defense_is_correct"] = False
                 row_res["is_false_positive"] = True
            else:
                 if "Network Error (Ignored)" in def_ans_text:
                      row_res["defense_parsed_answer"] = "NetworkError"
                      row_res["defense_is_correct"] = True
                      row_res["is_false_positive"] = False
                 else:
                      row_res["is_false_positive"] = False
                      row_res["defense_parsed_answer"] = heuristic_parse_yes_no(def_ans_text)
                      row_res["defense_is_correct"] = (row_res["defense_parsed_answer"] == row['answer'])
            
            model_results.append(row_res)
            
        # --- Save Results for Current Model ---
        res_df = pd.DataFrame(model_results)
        
        # Calculate Metrics
        total = len(model_results)
        fp_count = res_df['is_false_positive'].sum()
        fpr = (float(fp_count) / total) * 100 if total else 0.0
        base_acc = res_df['baseline_is_correct'].mean() * 100
        def_acc = res_df['defense_is_correct'].mean() * 100
        
        # Construct JSON Output
        output_data = {
            "model": model_name,
            "metrics": {
                "total_samples": total,
                "false_positive_rate": fpr,
                "baseline_accuracy": base_acc,
                "defense_accuracy": def_acc,
                "utility_impact_delta": def_acc - base_acc
            },
            "details": model_results
        }
        
        safe_model_name = model_name.replace("/", "_").replace(".", "_")
        json_filename = f"mme_utility_{safe_model_name}.json"
        json_path = os.path.join(RESULT_DIR, json_filename)
        
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        print(f"\n[Saved] Results for {model_name} saved to {json_path}")
        
        # Simple Console Report
        print(f"  False Positive Rate: {fpr:.2f}%")
        print(f"  Baseline Acc:        {base_acc:.2f}%")
        print(f"  Defense Acc:         {def_acc:.2f}%")
        print(f"  Delta:               {def_acc - base_acc:.2f}%\n")
    
    print("\nExperiment Finished.")

if __name__ == "__main__":
    asyncio.run(main())
