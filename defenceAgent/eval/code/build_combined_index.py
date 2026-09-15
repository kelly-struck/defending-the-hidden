import json
import os
from pathlib import Path
from defenceAgent.retrieval.hn_rag import tfidf_vectorize, save_index

def read_jsonl(path):
    texts = []
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return texts
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if "text" in obj:
                    texts.append(obj["text"])
            except:
                continue
    return texts

def main():
    # Define sources
    base_dir = Path(r"d:\AgentScope\defenceAgent")
    files_to_merge = [
        base_dir / "memory/hard_negatives.jsonl",
        base_dir / "memory/siuo_hard_negatives.jsonl"
    ]
    
    output_index = base_dir / "eval/index/hn_tfidf.json"
    
    print("Merging knowledge bases...")
    all_texts = []
    for p in files_to_merge:
        docs = read_jsonl(str(p))
        print(f"  - Loaded {len(docs)} documents from {p.name}")
        all_texts.extend(docs)
        
    print(f"Total documents: {len(all_texts)}")
    
    # Build Index
    print("Building TF-IDF index...")
    index = tfidf_vectorize(all_texts)
    
    # Save
    save_index(index, str(output_index))
    print(f"Successfully saved combined index to: {output_index}")
    print("The agent will now use this combined knowledge base automatically.")

if __name__ == "__main__":
    main()
