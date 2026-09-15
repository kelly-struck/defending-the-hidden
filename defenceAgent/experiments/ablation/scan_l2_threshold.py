import os
import sys

# Disable ChromaDB Telemetry to prevent timeouts
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import json
import base64
import pandas as pd
import numpy as np
from tqdm import tqdm
import io
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root

from defenceAgent.retrieval.rag_manager import RAGManager

# Configs
MME_PATH = r"d:\AgentScope\defenceAgent\statics\MME\data"
MMSAFETY_PATH = r"d:\AgentScope\defenceAgent\statics\MM-SafetyBench\data" 
# Expand categories to ensure we get 100 samples
ATTACK_CATEGORIES = [
    "Fraud", "HateSpeech", "Physical_Harm", "Illegal_Activitiy", 
    "Sex", "Malware_Generation", "Political_Lobbying", "EconomicHarm", 
    "Privacy_Violence", "Legal_Opinion", "Health_Consultation"
]

SAMPLE_COUNT = 100 # Samples per set (Benign vs Attack)

OUTPUT_FILE = r"results/threshold_scores.json"

def bytes_to_data_uri(img_bytes, mime_type="image/jpeg"):
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:{mime_type};base64,{b64}"

def get_benign_samples(n=50):
    samples = []
    files = [os.path.join(MME_PATH, f) for f in os.listdir(MME_PATH) if f.endswith('.parquet')]
    
    print(f"Loading Benign samples from MME ({len(files)} files)...")
    per_file = max(1, n // len(files))
    
    for f in files:
        if len(samples) >= n: break
        try:
            df = pd.read_parquet(f, columns=['question', 'image'])
            # MME images are dicts {'bytes': b'...'}
            for _, row in df.head(per_file).iterrows():
                try:
                    img_data = row['image']
                    if isinstance(img_data, dict) and 'bytes' in img_data:
                        img_bytes = img_data['bytes']
                    else:
                        continue # Skip malformed
                    
                    samples.append({
                        "text": row['question'],
                        "image_uri": bytes_to_data_uri(img_bytes),
                        "source": "benign_mme"
                    })
                except Exception as e:
                    pass
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return samples[:n]

def get_attack_samples(n=50):
    samples = []
    # Collect files from target categories
    target_files = []
    for cat in ATTACK_CATEGORIES:
        cat_dir = os.path.join(MMSAFETY_PATH, cat)
        if os.path.exists(cat_dir):
            target_files.extend([os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith('.parquet')])
            
    print(f"Loading Attack samples from MM-SafetyBench ({len(target_files)} files)...")
    if not target_files: return []

    per_file = max(1, n // len(target_files))
    
    for f in target_files:
        if len(samples) >= n: break
        try:
            df = pd.read_parquet(f) # Columns usually id, question, image
            for _, row in df.head(per_file).iterrows():
                try:
                    img_bytes = row['image'] # Direct bytes
                    if not isinstance(img_bytes, bytes): continue
                    
                    samples.append({
                        "text": row['question'],
                        "image_uri": bytes_to_data_uri(img_bytes),
                        "source": "attack_mmsafety"
                    })
                except Exception:
                    pass
        except Exception:
            pass
            
    return samples[:n]

def calculate_similarity(dist):
    # Logic from orchestrator.py: implied_sim = 1.0 - (dist * dist) / 2.0
    return 1.0 - (dist * dist) / 2.0

def main():
    # 1. Setup
    if not os.path.exists("results"): os.makedirs("results")
    
    print("Initializing RAG Manager (Loading ChromaDB)...")
    rag = RAGManager(persist_directory=r'd:\AgentScope\chroma_db')
    
    # 2. Get Data
    benign_set = get_benign_samples(SAMPLE_COUNT)
    attack_set = get_attack_samples(SAMPLE_COUNT)
    
    print(f"Collected {len(benign_set)} Benign samples and {len(attack_set)} Attack samples.")
    
    results = {
        "benign_scores": [],
        "attack_scores": []
    }
    
    # 3. Scan Benign
    print("Scanning Benign Set...")
    for s in tqdm(benign_set):
        try:
            # query_similarity(text, image_url, n_results=1)
            hits = rag.query_similarity(text=s['text'], image_url=s['image_uri'], n_results=1)
            if hits:
                score = calculate_similarity(hits[0]['distance'])
                results["benign_scores"].append(score)
            else:
                results["benign_scores"].append(0.0)
        except Exception as e:
            print(f"Error processing benign sample: {e}")

    # 4. Scan Attack
    print("Scanning Attack Set...")
    for s in tqdm(attack_set):
        try:
            hits = rag.query_similarity(text=s['text'], image_url=s['image_uri'], n_results=1)
            if hits:
                score = calculate_similarity(hits[0]['distance'])
                results["attack_scores"].append(score)
            else:
                results["attack_scores"].append(0.0)
        except Exception as e:
            print(f"Error processing attack sample: {e}")
            
    # 5. Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved scores to {OUTPUT_FILE}")
    
    # 6. Quick Analysis
    benign = np.array(results["benign_scores"])
    attack = np.array(results["attack_scores"])
    
    print("\n--- Summary Statistics ---")
    print(f"Benign Mean: {np.mean(benign):.4f} | Max: {np.max(benign):.4f} | Min: {np.min(benign):.4f}")
    print(f"Attack Mean: {np.mean(attack):.4f} | Max: {np.max(attack):.4f} | Min: {np.min(attack):.4f}")

    # Suggest Threshold
    # Create simple ranges
    for t in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        fp = np.sum(benign > t) / len(benign)
        tp = np.sum(attack > t) / len(attack)
        print(f"Threshold {t}: TPR (Recall) = {tp*100:.1f}%, FPR (False Alarm) = {fp*100:.1f}%")

if __name__ == "__main__":
    main()
