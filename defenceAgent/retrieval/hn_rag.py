from __future__ import annotations
"""
Lightweight TF‑IDF based RAG index for hard negative retrieval (pure Python).

Why: Avoid heavy deps; enable fast Top-K retrieval and tiny prompt injection.

Index format (JSON):
{
  "idf": { term: float, ... },
  "docs": [
     {"text": str, "vec": [[term, weight], ...], "norm": float}
  ],
  "meta": {"version": 1, "doc_count": int}
}

Usage:
  from defenceAgent.retrieval.hn_rag import build_index, save_index, load_index, query
  index = build_index(["text1", "text2", ...])
  save_index(index, "path.json")
  index = load_index("path.json")
  results = query(index, "user text", top_k=5)
"""
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+", re.UNICODE)

# Minimal bilingual bridge: map common Chinese risk phrases to English tokens
# This enables cross-lingual overlap when memory is English and query is Chinese.
BILINGUAL_SYNONYMS = {
    # finance
    "投资组合": ["portfolio"],
    "百分比配置": ["allocation", "percentages"],
    "买入时机": ["buy", "timing"],
    "收益": ["returns"],
    "收益率": ["returns", "roi"],
    "预测": ["predict", "prediction"],
    "高收益": ["high", "returns"],
    # health
    "剂量": ["dosage", "mg"],
    "处方": ["prescription"],
    "诊断": ["diagnose", "diagnosis"],
    # legal
    "诉讼": ["sue", "lawsuit"],
    "责任": ["liability"],
    "合同": ["contract"],
    # government / policy
    "政策": ["policy"],
    "分配": ["allocate", "allocation"],
    "监控": ["surveillance"],
    # politics
    "选民": ["voters"],
    "游说": ["lobby", "lobbying"],
    "竞选": ["campaign"],
    # gambling
    "赌博": ["gambling", "betting", "casino"],
    # automation / AI risk
    "无需人工": ["without", "human", "intervention"],
    "自动化": ["automate", "automation"],
}

def expand_query_with_synonyms(text: str) -> str:
    if not text:
        return text
    extra: List[str] = []
    for zh, ens in BILINGUAL_SYNONYMS.items():
        if zh in text:
            extra.extend(ens)
    if extra:
        return text + " " + " ".join(extra)
    return text

def tokenize(text: str) -> List[str]:
    # crude tokenizer: alnum and CJK unified
    return [t.lower() for t in TOKEN_RE.findall(text or "")]

def tfidf_vectorize(corpus: List[str]):
    N = len(corpus)
    docs_tf: List[Counter] = []
    df: Counter = Counter()
    for txt in corpus:
        toks = tokenize(txt)
        c = Counter(toks)
        docs_tf.append(c)
        for term in c:
            df[term] += 1
    idf: Dict[str, float] = {}
    for term, d in df.items():
        # add-one smoothing to avoid div-by-zero
        idf[term] = math.log((N + 1) / (d + 1)) + 1.0
    docs = []
    for i, c in enumerate(docs_tf):
        vec = {}
        for term, tf in c.items():
            vec[term] = (1 + math.log(tf)) * idf.get(term, 0.0)
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        # store as list for compactness
        docs.append({
            "text": corpus[i],
            "vec": [[t, v] for t, v in vec.items()],
            "norm": norm,
        })
    return {"idf": idf, "docs": docs, "meta": {"version": 1, "doc_count": N}}

def save_index(index: dict, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

def load_index(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def vectorize_query(index: dict, text: str):
    # Cross-lingual heuristic: append English synonyms if Chinese terms detected
    text = expand_query_with_synonyms(text)
    idf = index.get("idf", {})
    toks = tokenize(text)
    c = Counter(toks)
    qvec = {}
    for term, tf in c.items():
        qvec[term] = (1 + math.log(tf)) * idf.get(term, 0.0)
    qnorm = math.sqrt(sum(w * w for w in qvec.values())) or 1.0
    return qvec, qnorm

def dot_sparse(vec_list: List[Tuple[str, float]], qvec: Dict[str, float]) -> float:
    s = 0.0
    for t, v in vec_list:
        qv = qvec.get(t)
        if qv:
            s += v * qv
    return s

def query(index: dict, text: str, top_k: int = 5, min_len: int = 8):
    if not index or not index.get("docs"):
        return []
    qvec, qnorm = vectorize_query(index, text)
    if not qvec:
        return []
    scored = []
    for d in index["docs"]:
        if len(d.get("text", "")) < min_len:
            continue
        dot = dot_sparse(d["vec"], qvec)
        sim = dot / (d["norm"] * qnorm)
        scored.append((sim, d["text"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def summarize_for_prompt(matches: List[Tuple[float, str]], max_chars: int = 300) -> str:
    if not matches:
        return ""
    lines = []
    for score, text in matches:
        snip = text.strip().replace("\n", " ")
        if len(snip) > 160:
            snip = snip[:160] + "…"
        lines.append(f"- match({score:.2f}): {snip}")
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars] + "…"
    return out
