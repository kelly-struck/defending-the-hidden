from __future__ import annotations
"""Hard negative pattern memory & lightweight retrieval.

Design goals (v1):
 - Zero extra heavy dependencies (no faiss / sentence-transformers) for first iteration.
 - Provide simple lexical & token-overlap similarity scoring to surface previously missed (allowed) harmful queries.
 - Load patterns from a YAML rule file plus optional JSONL memory entries.

Rule file format (hard_negative_rules.yaml):
patterns:
  - id: finance_steps
    category: financial_advice
    action: revise
    template: "list steps to (invest|buy|sell|allocate|predict)"
    keywords: ["list steps", "predict", "high accuracy", "portfolio", "allocate", "buy", "sell"]
    rationale: "Executable / predictive financial guidance should not be directly allowed."

Memory JSONL (hard_negatives.jsonl) lines structure:
{"text": "user query text", "recommended_action": "block|revise", "pattern_id": "finance_steps"}

Scoring:
 - For each pattern, if any keyword substring exists -> base score += 1 per hit
 - Token Jaccard overlap between query tokens and template tokens -> added as fractional boost
 - Memory exact/near duplicate (Jaccard > 0.6) -> force include pattern

Returned retrieval object fields:
  {"pattern_id", "category", "recommended_action", "score", "rationale"}

This is intentionally simple; future versions can swap in embedding similarity transparently.
"""
import json
from pathlib import Path
from typing import List, Dict, Any
import yaml
import re

class HardNegativeMemory:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.rules_path = self.base_dir / "hard_negative_rules.yaml"
        self.memory_path = self.base_dir / "memory" / "hard_negatives.jsonl"
        self.patterns: List[Dict[str, Any]] = []
        self.memory_items: List[Dict[str, Any]] = []
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        # load patterns
        if self.rules_path.exists():
            try:
                data = yaml.safe_load(self.rules_path.read_text(encoding="utf-8")) or {}
                self.patterns = data.get("patterns", [])
            except Exception:
                self.patterns = []
        # load memory jsonl
        if self.memory_path.exists():
            try:
                for line in self.memory_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.memory_items.append(json.loads(line))
                    except Exception:
                        continue
            except Exception:
                pass
        self._loaded = True

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if t]

    def retrieve(self, query: str, topk: int = 5) -> List[Dict[str, Any]]:
        if not query:
            return []
        self.load()
        q_tokens = self._tokenize(query)
        q_set = set(q_tokens)
        scored: List[Dict[str, Any]] = []

        # memory duplicates boost
        for mem in self.memory_items:
            text = mem.get("text", "")
            if not text:
                continue
            m_tokens = self._tokenize(text)
            inter = len(q_set.intersection(m_tokens))
            union = len(q_set.union(m_tokens)) or 1
            jacc = inter / union
            if jacc >= 0.6:  # near duplicate of known hard negative
                scored.append({
                    "pattern_id": mem.get("pattern_id", "memory_item"),
                    "category": mem.get("category", "unknown"),
                    "recommended_action": mem.get("recommended_action", "revise"),
                    "score": 10.0,  # force top
                    "rationale": mem.get("rationale", "Similar to previously missed harmful query."),
                    "source": "memory_duplicate"
                })

        for pat in self.patterns:
            pid = pat.get("id")
            template = pat.get("template", "")
            keywords = pat.get("keywords", []) or []
            category = pat.get("category", "generic")
            rec_action = pat.get("action", "revise")
            rationale = pat.get("rationale", "")
            base = 0.0
            # keyword hits
            for kw in keywords:
                if kw.lower() in query.lower():
                    base += 1.0
            # token overlap with template
            t_tokens = self._tokenize(template)
            if t_tokens:
                inter = len(q_set.intersection(t_tokens))
                union = len(q_set.union(t_tokens)) or 1
                base += (inter / union) * 1.5  # fractional boost
            if base > 0:
                scored.append({
                    "pattern_id": pid,
                    "category": category,
                    "recommended_action": rec_action,
                    "score": round(base, 3),
                    "rationale": rationale,
                    "source": "rule_match"
                })
        # sort & truncate
        scored.sort(key=lambda x: x["score"], reverse=True)
        # deduplicate by pattern_id keeping highest score
        dedup: Dict[str, Dict[str, Any]] = {}
        for item in scored:
            pid = item.get("pattern_id")
            if pid not in dedup:
                dedup[pid] = item
        final = list(dedup.values())
        return final[:topk]

    def summarize_for_prompt(self, query: str, topk: int = 5) -> str:
        matches = self.retrieve(query, topk=topk)
        if not matches:
            return ""
        lines = ["Recent Hard-Negative Patterns matched (treat as high risk):"]
        for m in matches:
            lines.append(f"- [{m['pattern_id']}] cat={m['category']} action={m['recommended_action']} score={m['score']} :: {m['rationale']}")
        return "\n" + "\n".join(lines)

HNMemory = HardNegativeMemory()
