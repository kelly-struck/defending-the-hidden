from __future__ import annotations
from typing import Optional

from agentscope.message import Msg

from .models import ModelRegistry
from .safety_classifier import classify_with_details


class MLLMExecutor:
    """Call final MLLM according to judge action.

    If action is block -> return refusal string.
    If action is revise -> use judge.user_notice as system tip to rewrite.
    If action is allow -> pass through original user message.
    """

    def __init__(self, target_model_name: str = None) -> None:
        reg = ModelRegistry()
        if target_model_name:
            reg.target_model_name = target_model_name
        print(f"MLLMExecutor building model: {reg.target_model_name}")
        bundle = reg.build_target_mllm()
        self.model = bundle.model
        self.formatter = bundle.formatter

    async def __call__(self, user_msg: Msg, judge_meta: dict | None) -> Msg:
        action = (judge_meta or {}).get("action", "allow")
        if action == "block":
            return Msg(name="assistant", role="assistant", content=(judge_meta or {}).get("user_notice", "抱歉，该请求不符合使用政策。"))
        if action == "revise":
            sys_prompt = (judge_meta or {}).get("user_notice", "请将用户需求以合规、安全的方式改写并回答。")
            prompt = await self.formatter.format([
                Msg("system", sys_prompt, "system"),
                user_msg,
            ])
            resp = await self.model(prompt)
            # Preserve metadata (usage)
            meta = {"shadow_gate": True, "original": self._extract_text(resp.content), "classifier": None, "action": "revise"}
            if resp.metadata:
                meta.update(resp.metadata)
            return Msg(name="assistant", role="assistant", content=self._extract_text(resp.content), metadata=meta)
        # allow
        prompt = await self.formatter.format([
            Msg("system", "请直接、简洁地回答用户问题。", "system"),
            user_msg,
        ])
        resp = await self.model(prompt)
        answer_text = self._extract_text(resp.content)
        
        # Base metadata
        base_meta = {"shadow_gate": False}
        if resp.metadata:
            base_meta.update(resp.metadata)

        # Shadow safety gate (conditional): only trigger when存在明显风险信号，减少不必要调用和误杀
        jm = judge_meta or {}
        source = str(jm.get("source", ""))
        score = jm.get("score", 0)
        tags = jm.get("tags", []) or []
        reason = str(jm.get("reason", ""))
        risky_source = source in ("judge", "siuo_rules", "light_rules", "image", "text")
        risky_score = isinstance(score, (int, float)) and score >= 4
        risky_tags = isinstance(tags, list) and any(t for t in tags if str(t).lower() in ("siuo", "instruction", "risky_fusion", "environment-action", "semantic switch", "environmental risk"))
        risky_reason = ("risky" in reason.lower()) or ("self-harm" in reason.lower())
        should_classify = risky_source or risky_score or risky_tags or risky_reason
        if not should_classify:
            # 低风险 allow：直接返回，避免额外分类成本
            base_meta["shadow_skipped"] = True
            return Msg(name="assistant", role="assistant", content=answer_text, metadata=base_meta)

        try:
            is_unsafe, details = classify_with_details(answer_text)
        except Exception:
            is_unsafe, details = True, {"verdict": "unknown", "reasons": ["classifier_error"]}
        if is_unsafe:
            safe_reply = (
                "该回答可能存在安全风险。请保持安全与理性，如有情绪困扰可寻求专业帮助（示例：当地心理援助热线）。"
            )
            # 标记为需要改写/防御成功的等价结果，便于上游评估统计
            meta = {"shadow_gate": True, "original": answer_text, "classifier": details, "action": "revise"}
            # Merge usage info if available
            if resp.metadata:
                meta.update(resp.metadata)
            return Msg(name="assistant", role="assistant", content=safe_reply, metadata=meta)
        
        return Msg(name="assistant", role="assistant", content=answer_text, metadata=base_meta)

    def _extract_text(self, content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [c.get("text") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            return "\n".join([t for t in texts if t]) if texts else str(content)
        return str(content)

