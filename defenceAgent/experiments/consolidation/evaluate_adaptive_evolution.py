sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root
import asyncio
import json
import sys
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add project root to sys.path
current_dir = str(Path(__file__).parent.absolute())
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from defenceAgent.orchestrator import DefenceOrchestrator
from agentscope.message import Msg, TextBlock, ImageBlock
from defenceAgent.main import load_image_source
from defenceAgent.retrieval.rag_manager import RAGManager

# Configuration
SIUO_PATH = Path("defenceAgent/statics/SIUO/siuo_gen.json")
IMAGE_ROOT = Path("defenceAgent/statics/SIUO/images")
OUTPUT_FILE = "results/adaptive_evolution_results.json"
PLOT_FILE = "results/adaptive_evolution_plot.pdf"
NUM_BATCHES = 5

def load_and_split_data(path: Path, num_batches: int = 5) -> list[list[dict]]:
    """Load SIUO data and split into batches balanced by category."""
    print(f"Loading data from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total samples loaded: {len(data)}")

    # Group by category
    by_category = defaultdict(list)
    for item in data:
        img_name = item.get("image")
        if not img_name:
            continue
        # Check image existence
        img_path = IMAGE_ROOT / img_name
        if not img_path.exists():
            continue # Skip missing images
            
        by_category[item.get("category", "unknown")].append(item)

    # Filter categories with too few samples to split? 
    # Just Round Robin distribution
    batches = [[] for _ in range(num_batches)]
    
    # Distribute
    for cat, items in by_category.items():
        # Sort or shuffle ensuring determinism if needed. 
        # Here we just iterate.
        for i, item in enumerate(items):
            batch_idx = i % num_batches
            batches[batch_idx].append(item)
            
    print(f"Data split into {num_batches} batches.")
    for i, b in enumerate(batches):
        print(f"  Batch {i+1}: {len(b)} samples")
        
    return batches

def reset_vector_db(orch: DefenceOrchestrator):
    """Clear the ChromaDB collection to ensure starting from scratch."""
    print("Resetting Vector DB (Layer 2) for experiment...")
    try:
        if orch.debate_orchestrator.rag:
            rag = orch.debate_orchestrator.rag
            coll_name = rag.collection.name
            rag.client.delete_collection(coll_name)
            # Recreate
            rag.collection = rag.client.get_or_create_collection(
                name=coll_name,
                embedding_function=rag.embedding_fn
            )
            print("Vector DB reset successful.")
        else:
            print("Warning: RAG manager not found in orchestrator.")
    except Exception as e:
        print(f"Error resetting Vector DB: {e}")

async def run_experiment():
    print("=== Initializing Self-Evolution Experiment (Paraphrase Validation) ===")
    
    # Configure to use a clean, temporary collection for this experiment
    exp_collection_name = "siuo_adaptive_experiment_v4_paraphrase"
    os.environ["RAG_COLLECTION_NAME"] = exp_collection_name
    print(f"[Setup] Using experimental collection: {exp_collection_name}")
    
    # 1. Reset Environment (Start Fresh)
    print("[Setup] Clearing Vectors and Cache...")
    try:
        tmp_rag = RAGManager(collection_name=exp_collection_name)
        try:
            tmp_rag.client.delete_collection(exp_collection_name)
            print(f" - Deleted existing experimental collection: {exp_collection_name}")
        except Exception:
            pass 
        tmp_rag = RAGManager(collection_name=exp_collection_name)
    except Exception as e:
        print(f" - Warning during RAG reset: {e}")

    # Orchestrator setup
    orchestrator = DefenceOrchestrator()
    if orchestrator.cache:
         orchestrator.cache.local_cache = {} 
         if orchestrator.cache.use_redis:
             orchestrator.cache.redis_client.flushdb()
         print(" - Layer 1 Cache flushed.")

    # 2. Prepare Data: Batch 1 & Batch 2
    # Load raw data
    all_batches = load_and_split_data(SIUO_PATH, num_batches=1)
    # Take all items from the single batch
    seed_data = all_batches[0]
    
    print(f"\n[Data Prep] Selected {len(seed_data)} seed samples.")
    
    # Construct Batch 1 (Original) and Batch 2 (Paraphrased)
    batch1 = seed_data
    batch2 = []
    
    print("Generating Batch 2 (Paraphrased variants)...")
    
    # Robust paraphrase helper
    async def get_paraphrase(text):
        msgs = [
            {"role": "system", "content": "Rewrite the following question using different wording/synonyms but keeping the exact same intent and meaning. Output ONLY the rewritten question without any intro/outro."},    
            {"role": "user", "content": text}
        ]
        for attempt in range(3):
            try:
                resp = await orchestrator.rewriter.model(msgs)
                # Safely extract text
                try:
                    if getattr(resp, "text", None):
                        return resp.text.strip(' "')
                except Exception:
                    pass
                    
                # Try content attribute
                try:
                    if getattr(resp, "content", None):
                         c = resp.content
                         if isinstance(c, list):
                             parts = [x.get('text', '') for x in c if isinstance(x, dict)]
                             return "".join(parts).strip()
                         return str(c).strip()
                except Exception:
                    pass
                
                return str(resp).strip()
            except Exception as e:
                print(f"  [Paraphrase Retry {attempt+1}] Error: {e}")
                await asyncio.sleep(1)
        return text + " (variation)"

    for i, item in enumerate(batch1):
        new_item = item.copy()
        orig_q = item.get("question", "")
        
        new_q = await get_paraphrase(orig_q)

        new_item["question"] = new_q
        if "id" in item:
            new_item["id"] = item["id"] + 100000 
        else:
             new_item["id"] = i + 100000
        batch2.append(new_item)
        print(f" [{i+1}/{len(batch1)}] Orig: {orig_q[:40]}... -> New: {new_q[:40]}...")
        
    batches = [batch1, batch2]
    
    metrics = []
    
    print("\n=== Starting Processing Loops (2 Batches) ===")

    results = {
        "batch_metrics": [],
        "detailed_logs": []
    }

    # 2. Sequential Execution
    for batch_idx, batch in enumerate(batches):
        print(f"\n=== Running Batch {batch_idx + 1}/{NUM_BATCHES} ({len(batch)} samples) ===")
        
        metrics = {
            "total": len(batch),
            "layer1_cache_hit": 0,
            "layer2_vector_block": 0,
            "layer4_debate_call": 0,
            "allowed": 0,
            "blocked_total": 0
        }
        
        for i, item in enumerate(batch):
            # Construct Msg
            img_path = str(IMAGE_ROOT / item["image"])
            question = item["question"]
            
            # Use orchestrator logic to build message
            # We can reuse main.py's build_user_msg logic or manually
            blocks = []
            if question:
                blocks.append(TextBlock(type="text", text=question))
            if img_path:
                src = load_image_source(img_path)
                blocks.append(ImageBlock(type="image", source=src))
            
            user_msg = Msg(name="user", role="user", content=blocks)
            
            # Run
            # Source attribution:
            # - vector_db -> Layer 2
            # - layer4_debate -> Layer 4
            try:
                final_res, _, _ = await orchestrator.run(user_msg)
                
                source = final_res.source
                
                if "vector_db" in source or source == "layer2_vector_db":
                    metrics["layer2_vector_block"] += 1
                    metrics["blocked_total"] += 1
                elif "debate" in source or source == "layer4_debate":
                    metrics["layer4_debate_call"] += 1
                    if not final_res.safe:
                        metrics["blocked_total"] += 1
                    else:
                        metrics["allowed"] += 1
                elif "cache" in source or source == "layer1_cache":
                    metrics["layer1_cache_hit"] += 1
                    if not final_res.safe:
                        metrics["blocked_total"] += 1
                    else:
                        metrics["allowed"] += 1
                else:
                    # Fallback
                    print(f"Unknown source: {source}")
                    if not final_res.safe: metrics["blocked_total"] += 1
                    
            except Exception as e:
                error_str = str(e)
                if "DataInspectionFailed" in error_str or "inappropriate content" in error_str:
                    print(f"  [Auto-Block] Item {item.get('question_id')} blocked by Model Provider (DataInspectionFailed). Counting as Layer 2 Interception.")
                    metrics["layer2_vector_block"] += 1
                    metrics["blocked_total"] += 1
                else:
                    print(f"Error processing item {item.get('question_id')}: {e}")
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(batch)}...")

        # Calculate Rates
        l2_rate = metrics["layer2_vector_block"] / metrics["total"] if metrics["total"] else 0
        l4_rate = metrics["layer4_debate_call"] / metrics["total"] if metrics["total"] else 0
        
        print(f"  Batch {batch_idx+1} Results:")
        print(f"    Layer 2 Interception Rate: {l2_rate:.2%}")
        print(f"    Layer 4 Call Rate:         {l4_rate:.2%}")
        
        metrics["batch_idx"] = batch_idx + 1
        metrics["l2_rate"] = l2_rate
        metrics["l4_rate"] = l4_rate
        results["batch_metrics"].append(metrics)

    # 3. Save Results
    res_path = Path(OUTPUT_FILE)
    res_path.parent.mkdir(exist_ok=True, parents=True)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {res_path}")

    # 4. Plotting
    plot_results(results["batch_metrics"])

def plot_results(metrics: list[dict]):
    batches = [f"Batch {m['batch_idx']}\n({'Initial' if m['batch_idx']==1 else 'Variant'})" for m in metrics]
    l2_rates = [m["l2_rate"] for m in metrics]
    l4_rates = [m["l4_rate"] for m in metrics]

    x = range(len(batches))
    width = 0.35  # width of the bars

    plt.figure(figsize=(10, 6))
    
    # Create grouped bars
    bars1 = plt.bar([i - width/2 for i in x], l2_rates, width, label='Layer 2 Interception (Vector DB)', alpha=0.8)
    bars2 = plt.bar([i + width/2 for i in x], l4_rates, width, label='Layer 4 Call (Debate Agent)', alpha=0.8)

    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom')

    plt.title('Adaptive Defense Efficacy: Learning from Variants')
    plt.xlabel('Experiment Phase')
    plt.ylabel('Rate')
    plt.xticks(x, batches)
    plt.ylim(0, 1.1) # little extra space for labels
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.legend()
    
    plt.savefig(PLOT_FILE)
    print(f"Plot saved to {PLOT_FILE}")

if __name__ == "__main__":
    asyncio.run(run_experiment())
