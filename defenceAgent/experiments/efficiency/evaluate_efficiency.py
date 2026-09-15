sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root
import time
import json
import os
import random
import asyncio
import pandas as pd
import chromadb
from pathlib import Path
from agentscope.message import Msg, TextBlock, ImageBlock

# Imports from your project
from defenceAgent.orchestrator import DefenceOrchestrator
from defenceAgent.retrieval.rag_manager import RAGManager

# =================CONFIGURATION=================
DRY_RUN_LIMIT = 100         # Set to None for full run, specific integer (e.g., 2 or 5) for test run
DB_PATH = r'd:\AgentScope\chroma_db'
COLLECTION_NAME = 'harmbench_defense'
PARQUET_DIR = r"d:\AgentScope\defenceAgent\statics\MME\data" # Source for "Unknown" samples

# Pricing Config (RMB per 1k tokens - Estimate)
PRICING = {
    "embedding": 0.0018,  # Est. for embedding models
    "input": 0.001,       # Est. for Qwen3-VL-Plus input
    "output": 0.01       # Est. for Qwen3-VL-Plus output
}
# ===============================================

class EfficiencyEvaluator:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
        self.results = []
        
    def get_known_samples(self, limit=50):
        """Fetch samples associated with vector DB (Known Attacks)."""
        try:
            coll = self.chroma_client.get_collection(COLLECTION_NAME)
            # Fetch random items
            count = coll.count()
            if count == 0:
                print("Warning: Vector DB is empty.")
                return []
            
            fetch_limit = min(limit, count)
            # Retrieve actual documents (Text + Image URL stored in metadata usually)
            res = coll.get(limit=fetch_limit, include=['documents', 'metadatas'])
            
            samples = []
            for i in range(len(res['documents'])):
                doc = res['documents'][i]
                meta = res['metadatas'][i] if res['metadatas'] else {}
                
                # Try to clean doc text if it's the JSON dumped format from simple RAG
                text_content = doc
                # If your RAG stores JSON in document, parse it? 
                # Orchestrator stores case with text=... 
                # Let's assume raw text for now or simple string.
                
                sample = {
                    "type": "known",
                    "text": text_content, 
                    "image": meta.get("image_url", None), # Might be None
                    "id": res['ids'][i]
                }
                samples.append(sample)
                
            # If we need more than we have, cycle them
            if len(samples) < limit and len(samples) > 0:
                original_len = len(samples)
                while len(samples) < limit:
                    # Pick random valid sample to duplicate
                    base = samples[len(samples) % original_len]
                    # Create copy with modified ID to track
                    dup = base.copy()
                    dup['id'] = f"{base['id']}_dup_{len(samples)}"
                    samples.append(dup)
                    
            return samples[:limit]
        except Exception as e:
            print(f"Error fetching known samples: {e}")
            return []

    def get_unknown_samples(self, limit=50):
        """Fetch benign/new samples from MME parquet (Unknown)."""
        samples = []
        try:
            files = [os.path.join(PARQUET_DIR, f) for f in os.listdir(PARQUET_DIR) if f.endswith(".parquet")]
            if not files:
                return []
            
            random.shuffle(files) # Randomize files
            
            for f in files:
                if len(samples) >= limit:
                    break
                try:
                    df = pd.read_parquet(f)
                    # Sample what we need from this file
                    remaining = limit - len(samples)
                    
                    # Take random rows
                    take_n = min(len(df), remaining)
                    if take_n <= 0: continue
                    
                    sample_df = df.sample(n=take_n)
                    
                    for _, row in sample_df.iterrows():
                        import base64
                        img_b64 = None
                        if 'image' in row and isinstance(row['image'], dict) and 'bytes' in row['image']:
                            img_b64 = base64.b64encode(row['image']['bytes']).decode('utf-8')
                        
                        samples.append({
                            "type": "unknown",
                            "text": row['question'],
                            "image_b64": img_b64,
                            "id": str(row['question_id'])
                        })
                except Exception as file_err:
                    print(f"Skipping file {f}: {file_err}")
                    
            # Fallback fill
            if len(samples) < limit and len(samples) > 0:
                 original_len = len(samples)
                 while len(samples) < limit:
                    base = samples[len(samples) % original_len]
                    dup = base.copy()
                    dup['id'] = f"{base['id']}_dup_{len(samples)}"
                    samples.append(dup)

        except Exception as e:
            print(f"Error fetching unknown samples: {e}")
        return samples[:limit]

    def prepare_traffic(self, n=100):
        """
        Construct a traffic sequence for N samples.
        Target Distribution:
        - 20% Replay (Requires 20% Base Setup + 20% Replay calls)
          -> Actually, to get '20% Replay' in the logs, we run 20 replays.
          -> We use 'Known' samples as bases for Replay.
        - 40% Known (L2 hit) -> These are independent known samples.
        - 40% Unknown (L3 hit)
        
        Total distinct Base samples needed:
        - For Replay: 20 samples (Setup)
        - For Known: 40 samples
        - For Unknown: 40 samples
        
        Traffic Sequence Construction (Total lines = 20(Setup) + 40(Known) + 40(Unknown) + 20(Replay) = 120 lines?
        The prompt asks for "Average Latency" of the system.
        Usually we evaluate on a fixed test set size (e.g. 100 requests).
        If the input stream has 100 requests:
          - 20 are Replays (of previously seen items).
          - 40 are Known (seen in DB, but not in immediate cache? Or just standard known attacks).
          - 40 are Unknown.
        
        So the stream has 100 items. 
        But for Replay to work, the "History" must exist.
        If we start cold, we must inject the "Originals" first.
        If we count the "Originals" as part of the "Known" or "Unknown" allocation, we are fine.
        
        Plan for 100 requests:
        1. Select 20 samples to serve as Replay Bases. These will appear TWICE.
           - Appearance 1: Counts as "Known Attack" (L2 hit).
           - Appearance 2: Counts as "Replay Attack" (L1 hit).
        2. Select 20 samples to serve as independent Known Attacks (Appear ONCE).
           - Counts as "Known Attack".
        3. Select 40 samples to serve as Unknown (Appear ONCE).
           - Counts as "Unknown Sample".
           
        Total Requests:
        - 20 (Base for Replay) -> Label: "Known Attack"
        - 20 (Replay)          -> Label: "Replay Attack"
        - 20 (Independent Known)-> Label: "Known Attack"
        - 40 (Unknown)         -> Label: "Unknown Sample"
        Sum = 100 requests.
        
        Distribution Result:
        - Known Attack Lines: 20 + 20 = 40 (40%)
        - Replay Attack Lines: 20 (20%)
        - Unknown Sample Lines: 40 (40%)
        Perfect match.
        """
        
        # Calculate quantiles
        n_replay = int(n * 0.2)
        n_known_solo = int(n * 0.2) # Total known lines = 40, so 20 from base, 20 independent
        n_unknown = n - n_replay - n_replay - n_known_solo # Remainder (approx 40)
        
        # We generally treat "Known" samples as the pool for Replay
        # Need: n_replay (bases) + n_known_solo = total known samples needed
        total_known_needed = n_replay + n_known_solo
        
        known_pool = self.get_known_samples(limit=total_known_needed)
        unknown_pool = self.get_unknown_samples(limit=n_unknown)
        
        traffic = []
        
        # 1. Bases for Replay (Labelled as Known Attack initially)
        replay_bases = known_pool[:n_replay]
        for s in replay_bases:
            # First pass: It's a Known Attack targeting L2
            traffic.append({**s, "expected_layer": "L2", "label": "Known Attack"})
            
        # 2. Independent Known
        solo_known = known_pool[n_replay:]
        for s in solo_known:
             traffic.append({**s, "expected_layer": "L2", "label": "Known Attack"})
             
        # 3. Unknown
        for s in unknown_pool:
             traffic.append({**s, "expected_layer": "L3", "label": "Unknown Sample"})
             
        # 4. The Replays (Duplicates of Group 1)
        for s in replay_bases:
             traffic.append({**s, "expected_layer": "L1", "label": "Replay Attack"})
             
        # Order: 
        # Current list: [Bases... Solos... Unknowns... Replays...]
        # This order guarantees Bases run before Replays.
        # We can shuffle strictly within subsets or keep block structure.
        # Keeping block structure is safer to ensure cache is warm for Replays.
        # However, to simulate mixed traffic, we can interleave.
        # But we must ensure Base[i] comes before Replay[i].
        # Simple Block structure is acceptable for Efficiency Eval.
        
        return traffic

    async def run_single(self, orchestrator, sample, mode_name):
        """Run a single sample through orchestrator and measure."""
        
        # 1. Construct Message
        blocks = []
        if sample.get('image_b64'):
            blocks.append(ImageBlock(type="image", source={"type": "base64", "media_type": "image/jpeg", "data": sample['image_b64']}))
        elif sample.get('image'):
            # URL
            blocks.append(ImageBlock(type="image", source={"type": "url", "url": sample['image']}))
            
        blocks.append(TextBlock(type="text", text=sample['text']))
        msg = Msg(name="user", role="user", content=blocks)
        
        # 2. Run & Measure
        t0 = time.perf_counter()
        final_res, img_res, txt_res = await orchestrator.run(msg)
        t1 = time.perf_counter()
        
        latency_ms = (t1 - t0) * 1000
        
        # 3. Calculate Cost
        # L3 uses agents, orchestrator doesn't return total tokens directly unless we parse metadata
        # L1/L2 cost is mostly 0 or embedding.
        
        tokens_input = 0
        tokens_output = 0
        tokens_embedding = 0 
        
        source = final_res.source # layer1_cache, layer2_vector_db, layer4_debate
        
        # Extract Token Usage from Debate Transcript if available (L3)
        if hasattr(final_res.raw, "metadata"):
            meta = final_res.raw.metadata
            if meta and "total_tokens" in meta:
                 # This is a simplified total from our updated orchestrator_debate
                 # Assume 70% input, 30% output split for rough cost calc if detailed breakdown not avail
                 total_t = meta["total_tokens"]
                 # Usually debate is heavy on input (history)
                 tokens_input = int(total_t * 0.8)
                 tokens_output = int(total_t * 0.2)
        
        # If Layer 2, we incurred embedding cost
        # We need to estimate embedding tokens (approx chars / 4)
        # Note: "vector_db" is the source returned by orchestrator for L2 hits
        # "layer4_debate" means it passed L2 (so embedding ran) and went to L3
        if source in ["vector_db", "layer2_vector_db", "layer4_debate"]:
             # Passed L1, so ran embedding
             est_tokens = len(str(sample.get('text', ''))) // 3
             # Add estimate for image tokenization in embedding model (e.g. ~1000 tokens)
             if sample.get('image_b64') or sample.get('image'):
                 est_tokens += 1000 
             tokens_embedding = est_tokens
        
        cost = (tokens_embedding / 1000 * PRICING['embedding']) + \
               (tokens_input / 1000 * PRICING['input']) + \
               (tokens_output / 1000 * PRICING['output'])
               
        return {
            "mode": mode_name,
            "type": sample.get('label'),
            "expected_layer": sample.get('expected_layer'),
            "actual_layer": source,
            "latency_ms": latency_ms,
            "tokens_total": tokens_input + tokens_output + tokens_embedding,
            "cost_rmb": cost
        }

    async def run_experiment(self):
        print(f"Preparing traffic (Target Limit: {DRY_RUN_LIMIT})...")
        traffic = self.prepare_traffic(n=DRY_RUN_LIMIT)
        print(f"Traffic prepared: {len(traffic)} samples.")
        
        # --- Mode 1: Pure Debate (Baseline) ---
        print("\n=== Running Mode: Pure Debate (Layer 3 Only) ===")
        # Hack to disable L1/L2: Instantiate orchestrator and nullify props
        orch_base = DefenceOrchestrator()
        orch_base.cache = None # Disable L1
        orch_base.verify_with_vector_db = lambda *args, **kwargs: None # Disable L2 check
        
        for i, sample in enumerate(traffic):
            print(f"  Sample {i+1}/{len(traffic)} ({sample['label']}) running...")
            res = await self.run_single(orch_base, sample, "Pure Debate")
            self.results.append(res)
            print(f"    -> {res['actual_layer']} | {res['latency_ms']:.1f}ms | ¥{res['cost_rmb']:.5f}")

        # --- Mode 2: Cascade Defense (Ours) ---
        print("\n=== Running Mode: Cascade Defense (L1 + L2 + L3) ===")
        # Standard Orchestrator
        orch_cascade = DefenceOrchestrator()
        
        # PRE-WARM FOR REPLAY: 
        # For the replay attack to match L1, the First "Known" sample must be in cache.
        # But `prepare_traffic` sends k1 then k1.
        # In `Pure Debate`, nothing enters cache.
        # In `Cascade Defense`, the first k1 will run. It might hit L2 or L3.
        # Then it will be saved to Cache.
        # Then the second k1 (Replay) should hit L1.
        # So natural execution order is sufficient.
        
        # Note: We need to clear cache before starting Cascade run to ensure fair "Replay" test?
        # Or we assume system starts fresh.
        if orch_cascade.cache and getattr(orch_cascade.cache, 'use_redis', False):
             orch_cascade.cache.redis_client.flushdb()
             orch_cascade.cache.local_cache = {}
             print("  [Setup] Cleared L1 Cache.")
        elif orch_cascade.cache:
             orch_cascade.cache.local_cache = {}
             print("  [Setup] Cleared L1 Local Cache.")

        for i, sample in enumerate(traffic):
            print(f"  Sample {i+1}/{len(traffic)} ({sample['label']}) running...")
            res = await self.run_single(orch_cascade, sample, "Cascade Defense")
            self.results.append(res)
            print(f"    -> {res['actual_layer']} | {res['latency_ms']:.1f}ms | ¥{res['cost_rmb']:.5f}")
            
    def save_report(self):
        df = pd.DataFrame(self.results)
        print("\n=== Final Report ===")
        print(df.groupby(['mode', 'type'])[['latency_ms', 'cost_rmb']].mean())
        
        os.makedirs("results", exist_ok=True)
        file_path = "results/efficiency_experiment.csv"
        df.to_csv(file_path, index=False)
        print(f"\nDetailed results saved to {file_path}")

if __name__ == "__main__":
    evaluator = EfficiencyEvaluator()
    asyncio.run(evaluator.run_experiment())
    evaluator.save_report()
