from __future__ import annotations
import asyncio
from typing import Optional, Tuple
from dataclasses import dataclass
import time

from agentscope.message import Msg, TextBlock, ImageBlock

from .agents import ImageSafetyAgent, TextSafetyAgent, JudgeAgent, RewriterAgent, DebateAttackerAgent
from .orchestrator_debate import DebateOrchestrator
from .defense_cache import DefenseCache
from .light_rules import precheck_text, precheck_cross_modal
import json
from .visual_tags import get_visual_tags
from .ocr_utils import ocr_text_from_msg_content_blocks
from .guard_kb import extract_plain_text
from pathlib import Path

# RAG imports
try:
    from .retrieval.hn_rag import load_index as _hn_load_index, query as _hn_query
    _HN_RAG_PATH = (Path(__file__).parent / "eval" / "index" / "hn_tfidf.json").as_posix()
    _HN_RAG_INDEX = None
    try:
        p = Path(_HN_RAG_PATH)
        if p.exists():
            _HN_RAG_INDEX = _hn_load_index(_HN_RAG_PATH)
    except Exception:
        _HN_RAG_INDEX = None
except Exception:
    _HN_RAG_INDEX = None


@dataclass
class SafetyResult:
    safe: bool
    source: str  # "image" | "text" | "judge"
    raw: Msg
    score: float = 0.0
    confidence: float = 1.0


class DefenceOrchestrator:
    def __init__(self,
                 fast_mode: bool = False,
                 skip_judge_on_low_risk: bool = False,
                 visual_tags_on_hint: bool = False,
                 max_visual_tags_calls: int = 1,
                 siuo_text_hints: Optional[list[str]] = None) -> None:
        self.image_agent = ImageSafetyAgent()
        self.text_agent = TextSafetyAgent()
        self.judge = JudgeAgent()
        self.rewriter = RewriterAgent()
        self.attacker = DebateAttackerAgent()
        self.debate_orchestrator = DebateOrchestrator()
        self.cache = DefenseCache()
        self.fast_mode = fast_mode
        self.skip_judge_on_low_risk = skip_judge_on_low_risk
        self.visual_tags_on_hint = visual_tags_on_hint
        self.max_visual_tags_calls = max_visual_tags_calls
        # 仅在需要启用 hint 触发时使用（与 light_rules 中保持一致即可）
        self._siuo_hints = siuo_text_hints or [
            "encouragement", "poem", "farewell", "new world", "relieve", "relieve stress",
            "head bang", "headbang", "lie down", "sink", "with me to the sea",
            "pills", "medication", "overdose", "camera", "recording", "hidden",
            "religion", "holy site", "singing", "playlist", "brush", "paint",
            "头磕", "跳下", "躺在", "海里", "药", "安眠药", "摄像头", "偷拍"
        ]

    async def _check_image(self, user_msg: Msg) -> SafetyResult:
        # 仅保留图片块
        img_blocks = user_msg.get_content_blocks("image") if hasattr(user_msg, "get_content_blocks") else []
        if not img_blocks:
            # 没有图片即视为对图片安全检查通过
            dummy = Msg(
                name="image_safety",
                role="assistant",
                content='{"action":"allow","score":0,"confidence":1.0}',
                metadata={"action": "allow", "score": 0, "confidence": 1.0},
            )
            return SafetyResult(safe=True, source="image", raw=dummy, score=0.0, confidence=1.0)
        
        # 先进行基于 OCR 的轻规则预检（若可用）：如图中文字含越狱提示则短路
        try:
            ocr_text = ocr_text_from_msg_content_blocks(img_blocks, max_images=3)
        except Exception:
            ocr_text = None
        if ocr_text:
            hit = precheck_text(ocr_text)
            if hit is not None:
                if hit.action == "hard_block":
                    meta = {
                        "action": "block",
                        "source": "light_rules_ocr",
                        "matched_rule": hit.rule,
                        "reason": hit.reason,
                        "tags": hit.tags,
                        "ocr_preview": (ocr_text[:200] + ("…" if len(ocr_text) > 200 else "")),
                    }
                    raw = Msg(name="light_rules_ocr", role="assistant", content=str(meta), metadata=meta)
                    return SafetyResult(safe=False, source="rules", raw=raw, score=10.0, confidence=1.0)
                else:
                    # soft → 触发改写，统一标记为 revise
                    try:
                        rw = await self.rewriter(Msg(name="user", role="user", content=[TextBlock(type="text", text=ocr_text)]))
                        rw_meta = rw.metadata or {}
                        revised_text = (rw_meta or {}).get("revised_text") if isinstance(rw_meta, dict) else None
                    except Exception:
                        revised_text = None
                    meta = {
                        "action": "revise",
                        "source": "light_rules_ocr",
                        "matched_rule": hit.rule,
                        "reason": hit.reason,
                        "tags": hit.tags,
                        "revised_text": revised_text or "Please provide a safe, high-level description of visible text in the image.",
                        "ocr_preview": (ocr_text[:200] + ("…" if len(ocr_text) > 200 else "")),
                    }
                    raw = Msg(name="light_rules_ocr", role="assistant", content=str(meta), metadata=meta)
                    return SafetyResult(safe=False, source="rules", raw=raw, score=5.0, confidence=0.8)

        img_only_msg = Msg(name=user_msg.name, role=user_msg.role, content=img_blocks)
        res = await self.image_agent(img_only_msg)
        meta = res.metadata or {}
        action = None
        score = 0.0
        confidence = 1.0
        if isinstance(meta, dict):
            action = meta.get("action")
            score = float(meta.get("score", 0))
            confidence = float(meta.get("confidence", 1.0))
            if not action and "verdict" in meta:
                action = "allow" if meta.get("verdict") == "safe" else "block"
        is_allow = (action == "allow")
        return SafetyResult(safe=is_allow, source="image", raw=res, score=score, confidence=confidence)

    async def _check_text(self, user_msg: Msg) -> SafetyResult:
        # 仅保留文本块
        text_blocks = user_msg.get_content_blocks("text") if hasattr(user_msg, "get_content_blocks") else []
        if not text_blocks:
            dummy = Msg(
                name="text_safety",
                role="assistant",
                content='{"action":"allow","score":0,"confidence":1.0}',
                metadata={"action": "allow", "score": 0, "confidence": 1.0},
            )
            return SafetyResult(safe=True, source="text", raw=dummy, score=0.0, confidence=1.0)
        txt_only_msg = Msg(name=user_msg.name, role=user_msg.role, content=text_blocks)
        res = await self.text_agent(txt_only_msg)
        meta = res.metadata or {}
        action = None
        score = 0.0
        confidence = 1.0
        if isinstance(meta, dict):
            action = meta.get("action")
            score = float(meta.get("score", 0))
            confidence = float(meta.get("confidence", 1.0))
            if not action and "verdict" in meta:
                action = "allow" if meta.get("verdict") == "safe" else "block"
        is_allow = (action == "allow")
        return SafetyResult(safe=is_allow, source="text", raw=res, score=score, confidence=confidence)

    async def _run_debate(self, user_msg: Msg, img_res: SafetyResult, txt_res: SafetyResult) -> Msg:
        """Run a dynamic 2-round debate using Layer 4 DebateOrchestrator."""
        
        # We delegate to the specialized DebateOrchestrator
        # It handles the full debate logic (Defender -> Attacker -> Judge [-> Rebuttal -> Judge])
        # and returns a dict with verdict.
        
        res = await self.debate_orchestrator.run(input_msg=user_msg)
        
        verdict = res.get("verdict", "unsafe")
        transcript = res.get("debate_transcript", {})
        total_tokens = res.get("total_tokens", 0)
        
        # Map verdict to action for safety result
        action = "allow" if verdict == "safe" else "block"
        if verdict == "conditional_unsafe":
            action = "block" # Or potentially "revise" if we support conditional output
        
        # Construct result Msg
        # We put the detailed result in content and metadata
        content_str = json.dumps(res, indent=2, ensure_ascii=False)
        
        meta = {
            "action": action,
            "verdict": verdict,
            "score": 0.0 if verdict == "safe" else 10.0,
            "confidence": 1.0,
            "debate_transcript": transcript,
            "total_tokens": total_tokens,
            "debate_occurred": True
        }
        
        return Msg(name="Judge", role="assistant", content=content_str, metadata=meta)


    def verify_with_vector_db(self, text_query: str, image_source: Optional[str] = None, threshold: float = 0.85) -> SafetyResult | None:
        """
        Layer 2: Check if text_query (and optional image) is similar to known harmful queries in RAG.
        Uses Multimodal Embedding matching.
        """
        # If no RAG available or no query text, skip
        if not self.debate_orchestrator.rag or not text_query:
            return None
            
        try:
            rag = self.debate_orchestrator.rag
            # Use the new query_similarity interface supporting multimodal
            # image_source: can be url or base64 or path
            hits = rag.query_similarity(text=text_query, image_url=image_source, n_results=1)
            
            if not hits:
                return None
                
            top_hit = hits[0]
            dist = top_hit["distance"] # L2 distance usually
            
            # Calculate implied similarity from L2 distance assuming unit vectors
            # sim = 1 - (dist^2)/2
            # Threshold > 0.8
            implied_sim = 1.0 - (dist * dist) / 2.0
            
            if implied_sim > threshold:
                matched_doc = top_hit["document"]
                # Try to parse matched doc if it is JSON (multimodal)
                try:
                    import json
                    doc_data = json.loads(matched_doc)
                    if isinstance(doc_data, dict) and "text" in doc_data:
                        matched_snippet = f"Text: {doc_data.get('text', '')[:100]}..."
                    else:
                        matched_snippet = matched_doc[:200]
                except:
                    matched_snippet = matched_doc[:200]

                meta = {
                    "action": "block",
                    "source": "vector_db",
                    "reason": f"High similarity ({implied_sim:.2f}) to known harmful sample.",
                    "matched_sample": matched_snippet
                }
                raw = Msg(name="vector_db", role="assistant", content=str(meta), metadata=meta)
                return SafetyResult(safe=False, source="vector_db", raw=raw, score=10.0, confidence=1.0)
                
        except Exception as e:
            print(f"Vector DB Check Error: {e}")
            
        return None

    async def run(self, user_msg_with_modalities: Msg) -> Tuple[SafetyResult, Optional[SafetyResult], Optional[SafetyResult]]:
        """Return tuple: (final, image_result, text_result).

        Flow (Updated):
        Layer 1: Memory (Redis/Hash Cache)
        Layer 2: Vector DB (Similarity Check)
        Layer 3: (Skipped/Base Agents currently unused for decision, only for logging if needed)
        Layer 4: Debate (Agentscope Multi-Agent)
           - Judge decides
           - Feedback Loop -> Vector DB
        """
        perf = {"t0": time.perf_counter()}
        
        # Extract text and image for cache keys
        plain_text = extract_plain_text(user_msg_with_modalities)
        img_blocks = user_msg_with_modalities.get_content_blocks("image") if hasattr(user_msg_with_modalities, "get_content_blocks") else []
        # Construct image path/data for hash. 
        # For simplicity, if multiple images, we hash them all or take first.
        image_key = None
        image_source_for_rag = None
        
        if img_blocks:
            src = img_blocks[0].get("source", "")
            image_key = str(src)
            
            # Extract valid image source for RAG (url or base64)
            if isinstance(src, dict):
                 if src.get("type") == "url":
                     image_source_for_rag = src.get("url")
                 elif src.get("type") == "base64":
                     image_source_for_rag = f"data:{src.get('media_type')};base64,{src.get('data')}"
            elif hasattr(src, "url") and src.url:
                 image_source_for_rag = src.url
            elif hasattr(src, "data") and src.data:
                 mt = getattr(src, "media_type", "image/jpeg")
                 image_source_for_rag = f"data:{mt};base64,{src.data}"
            elif isinstance(src, str) and (src.startswith("http") or src.startswith("data:")):
                 image_source_for_rag = src
            
        # ====================
        # Layer 1: Memory Cache (Redis/Hash)
        # ====================
        if self.cache:
            cache_hit = self.cache.get(image_key, plain_text)
            if cache_hit:
                # Hit! Return cached result.
                # cache_hit is a dict { "safe": bool, "verdict": ... }
                is_safe = cache_hit.get("safe", False)
                action = "allow" if is_safe else "block"
                
                # Reconstruct a pseudo-result
                raw_msg = Msg(name="DefenseCache", role="assistant", content=f"Cached Result: {action}", metadata=cache_hit)
                
                final_res = SafetyResult(
                    safe=is_safe,
                    source="layer1_cache",
                    raw=raw_msg,
                    score=0.0 if is_safe else 10.0
                )
                return final_res, None, None

        # ====================
        # Layer 2: Vector DB Check
        # ====================
        # Check text AND image similarity using Multimodal Embedding.
        
        vect_res = self.verify_with_vector_db(plain_text, image_source=image_source_for_rag, threshold=0.85)
        if vect_res:
             # Hit! High similarity to bad sample -> Block
             if self.cache:
                self.cache.set(image_key, plain_text, {
                    "safe": False, 
                    "verdict": "unsafe", 
                    "source": "layer2_vector_db",
                    "reason": vect_res.raw.metadata.get("reason")
                })
             return vect_res, None, None


        # ====================
        # Layer 4: Debate (The Main Framework)
        # ====================
        
        # Only if Layer 1 & 2 missed.
        
        # Run Debate Orchestrator
        judge_msg = await self._run_debate(user_msg_with_modalities, None, None) # img_res/txt_res not needed/available if we skip Layer 2 base agents
        
        # Parse Judge Result
        meta = judge_msg.metadata or {}
        action = meta.get("action") 
        verdict = meta.get("verdict", "unsafe")
        
        is_safe = (verdict == "safe")
        
        final_res = SafetyResult(
            safe=is_safe, 
            source="layer4_debate", 
            raw=judge_msg, 
            score=0.0 if is_safe else 10.0
        )
        
        # ====================
        # Post-Processing: Cache & Loop
        # ====================
        
        # 1. Update Layer 1 Cache
        if self.cache:
            self.cache.set(image_key, plain_text, {
                "safe": is_safe,
                "verdict": verdict,
                "source": "layer4_debate",
                "reason":  meta.get("debate_transcript", {}).get("final_decision", "")
            })

        # 2. Update Layer 2 Vector DB (Feedback Loop)
        # This is already handled INSIDE DebateOrchestrator.run() 
        # as per previous instruction ("Learning Step").
        # So we don't need to do it here again.
        
        return final_res, None, None
