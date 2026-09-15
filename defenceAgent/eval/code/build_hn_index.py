from __future__ import annotations
"""Build TF-IDF RAG index from hard_negatives.jsonl for fast retrieval.

Usage:
  python -m defenceAgent.eval.code.build_hn_index \
    --memory_path defenceAgent/memory/hard_negatives.jsonl \
    --out_index defenceAgent/eval/index/hn_tfidf.json

This reads `text` from each JSONL line and builds a compact cosine-sim index.
"""
import argparse
import json
from pathlib import Path
from typing import List

from defenceAgent.retrieval.hn_rag import tfidf_vectorize, save_index


def read_memory_jsonl(path: str) -> List[str]:
    texts = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("text")
            if t and isinstance(t, str):
                texts.append(t)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory_path", required=True)
    ap.add_argument("--out_index", required=True)
    args = ap.parse_args()

    texts = read_memory_jsonl(args.memory_path)
    index = tfidf_vectorize(texts)
    save_index(index, args.out_index)
    print(json.dumps({
        "built_docs": len(texts),
        "index_path": args.out_index
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
