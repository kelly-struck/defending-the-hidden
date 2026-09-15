from __future__ import annotations
import json
import base64
from pathlib import Path
from typing import Optional
from agentscope.message import Msg, URLSource, Base64Source
from .agents import DebateJudgeAgent, DebateAttackerAgent, DebateDefenderAgent


def load_image_source(path_or_url: str):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return URLSource(type="url", url=path_or_url)
    p = Path(path_or_url)
    if not p.exists():
        if path_or_url.startswith("file://"):
            return URLSource(type="url", url=path_or_url)
        raise FileNotFoundError(f"Image file not found: {path_or_url}")
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    mt = "image/png" if p.suffix.lower() in {".png"} else "image/jpeg"
    return Base64Source(type="base64", media_type=mt, data=b64)


class JudgeOnlyOrchestrator:
    """w/o-Attacker Agent: 只保留Judge Agent直接裁决"""
    def __init__(self):
        self.judge = DebateJudgeAgent()

    async def run(self, image_path: str, text_query: str) -> dict:
        content = []
        if image_path:
            source = load_image_source(image_path)
            if source["type"] == "url":
                src_dict = {"type": "url", "url": source["url"]}
            else:
                src_dict = {"type": "base64", "media_type": source["media_type"], "data": source["data"]}
            content.append({"type": "image", "source": src_dict})
        if text_query:
            content.append({"type": "text", "text": text_query})
        user_msg = Msg(name="User", role="user", content=content)
        await self.judge.memory.clear()
        await self.judge.memory.add(user_msg)
        judge_msg = await self.judge(None)
        try:
            res_text = judge_msg.content
            if "```json" in res_text:
                json_str = res_text.split("```json")[1].split("```")[0]
            elif "```" in res_text:
                json_str = res_text.split("```")[1].split("```")[0]
            else:
                json_str = res_text
            data = json.loads(json_str.strip())
            return data
        except Exception as e:
            return {
                "verdict": "error",
                "reason": f"Failed to parse judge output: {str(e)}",
                "raw_output": judge_msg.content
            }


class SingleAgentControlOrchestrator:
    """R1-C1 / R2-C20 控制 arm：单 agent 承载非对称裁决规则 + 攻击方论证压力。

    设计（Phase 1 E4）：
    - Judge 复用 DebateJudgeAgent(symmetric=False)，system prompt 与全系统
      DebateOrchestrator 中的 Judge 提示**逐字一致**（同一 `_asymmetric_prompt()`），
      唯一差异是**没有独立的 Attacker/Defender 角色分工**（无多轮论辩 transcript）。
    - 非对称规则本身要求 Judge 主动采用最坏情况解读（"IF DOUBT EXISTS, BLOCK IT"、
      "Devil's Advocate Check"、Intent-as-Action），即"攻击方论证压力"已编码在规则中，
      因此单 agent 承载了与全系统相同的提示/规则/token 预算（max_tokens 一致）。

    Diff vs Full System（DebateOrchestrator）：
      - system prompt : 相同（逐字复用 _asymmetric_prompt()）
      - 输入          : 原始 image+text（无 Defender opening / Attacker opening / Rebuttal）
      - 角色分工      : 1 个 agent（无 Attacker/Defender 独立角色）
      - 返回 schema   : {verdict, debate_transcript, total_tokens, response} 与消融 arm 一致
    """
    def __init__(self):
        self.judge = DebateJudgeAgent(symmetric=False)

    async def run(self, image_path: str = None, text_query: str = None, input_msg: Msg = None) -> dict:
        if input_msg is not None:
            user_msg = input_msg
        else:
            content = []
            if image_path:
                source = load_image_source(image_path)
                if source["type"] == "url":
                    src_dict = {"type": "url", "url": source["url"]}
                else:
                    src_dict = {"type": "base64", "media_type": source["media_type"], "data": source["data"]}
                content.append({"type": "image", "source": src_dict})
            if text_query:
                content.append({"type": "text", "text": text_query})
            user_msg = Msg(name="User", role="user", content=content)

        await self.judge.memory.clear()
        await self.judge.memory.add(user_msg)
        judge_msg = await self.judge(None)

        total_tokens = 0
        if judge_msg.metadata and 'token_usage' in judge_msg.metadata:
            usage = judge_msg.metadata['token_usage']
            if isinstance(usage, dict):
                total_tokens += usage.get('total_tokens', 0)
            else:
                total_tokens += getattr(usage, 'total_tokens', 0)

        try:
            res_text = judge_msg.content
            if "```json" in res_text:
                json_str = res_text.split("```json")[1].split("```")[0]
            elif "```" in res_text:
                json_str = res_text.split("```")[1].split("```")[0]
            else:
                json_str = res_text
            data = json.loads(json_str.strip())
            if not isinstance(data, dict) or "verdict" not in data:
                raise ValueError("no verdict in parsed JSON")
        except Exception as e:
            data = {
                "verdict": "error",
                "reason": f"Failed to parse judge output: {str(e)}",
                "raw_output": judge_msg.content,
            }

        data["debate_transcript"] = {
            "single_agent_judge_output": judge_msg.content,
        }
        data["total_tokens"] = total_tokens
        data["response"] = judge_msg.content
        return data


class JudgeDefenderOrchestrator:
    """w/o-Attacker Agent: 保留Defense + Judge"""
    def __init__(self):
        self.defender = DebateDefenderAgent()
        self.judge = DebateJudgeAgent()

    async def run(self, image_path: str = None, text_query: str = None, input_msg: Msg = None) -> dict:
        extracted_text_query = text_query
        if input_msg:
            user_msg = input_msg
            content = user_msg.content
            if not isinstance(content, list):
                content = [{"type": "text", "text": str(content)}]
            if not extracted_text_query:
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        extracted_text_query = block.get("text")
                        break
                    elif hasattr(block, "type") and block.type == "text":
                        extracted_text_query = block.text
                        break
        else:
            content = []
            if image_path:
                source = load_image_source(image_path)
                if source["type"] == "url":
                    src_dict = {"type": "url", "url": source["url"]}
                else:
                    src_dict = {"type": "base64", "media_type": source["media_type"], "data": source["data"]}
                content.append({"type": "image", "source": src_dict})
            if text_query:
                content.append({"type": "text", "text": text_query})
            user_msg = Msg(name="User", role="user", content=content)

        await self.defender.memory.clear()
        await self.judge.memory.clear()
        total_tokens = 0

        def add_usage(msg):
            nonlocal total_tokens
            if msg.metadata and 'token_usage' in msg.metadata:
                usage = msg.metadata['token_usage']
                if isinstance(usage, dict):
                    total_tokens += usage.get('total_tokens', 0)
                else:
                    total_tokens += getattr(usage, 'total_tokens', 0)

        # 1. Defender speaks
        await self.defender.memory.add(user_msg)
        defender_opening_msg = await self.defender(None)
        add_usage(defender_opening_msg)
        defender_arg = defender_opening_msg.content

        # 2. Judge makes final verdict
        transcript = (
            f"--- DEBATE TRANSCRIPT (Ablation: No Prosecutor) ---\n"
            f"  Defender (Opening): {defender_arg}\n"
            f"------------------------------------\n"
            f"Make your final judgment. There is no Prosecutor. Evaluate if the Defender's logic holds. Output verdict: 'safe' or 'unsafe'."
        )
        judge_content = list(content)
        judge_content.append({"type": "text", "text": "\n\n" + transcript})
        judge_msg = Msg(name="User", role="user", content=judge_content)
        await self.judge.memory.add(judge_msg)
        judge_verdict_msg = await self.judge(None)
        add_usage(judge_verdict_msg)

        data = {}
        try:
            res_text = judge_verdict_msg.content
            if "```json" in res_text:
                json_str = res_text.split("```json")[1].split("```")[0]
            elif "```" in res_text:
                json_str = res_text.split("```")[1].split("```")[0]
            else:
                json_str = res_text
            data = json.loads(json_str.strip())
        except:
            data = {"verdict": "unsafe"}

        if data.get("verdict") not in ["safe", "unsafe"]:
            data["verdict"] = "unsafe"

        data["debate_transcript"] = {
            "round_1_defender": defender_arg,
            "round_1_attacker": None,
            "round_1_judge_verdict": data.get("verdict"),
            "round_1_judge_reason": data.get("reason", "Parse fallback")
        }
        data["total_tokens"] = total_tokens
        return data


class JudgeAttackerOrchestrator:
    """w/o-Defender Agent: 只保留Attack + Judge
    Attack Agent先发言，Judge直接裁决
    """
    def __init__(self):
        self.attacker = DebateAttackerAgent()
        self.judge = DebateJudgeAgent()

    async def run(self, image_path: str = None, text_query: str = None, input_msg: Msg = None) -> dict:
        extracted_text_query = text_query
        if input_msg:
            user_msg = input_msg
            content = user_msg.content
            if not isinstance(content, list):
                content = [{"type": "text", "text": str(content)}]
            if not extracted_text_query:
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        extracted_text_query = block.get("text")
                        break
                    elif hasattr(block, "type") and block.type == "text":
                        extracted_text_query = block.text
                        break
        else:
            content = []
            if image_path:
                source = load_image_source(image_path)
                if source["type"] == "url":
                    src_dict = {"type": "url", "url": source["url"]}
                else:
                    src_dict = {"type": "base64", "media_type": source["media_type"], "data": source["data"]}
                content.append({"type": "image", "source": src_dict})
            if text_query:
                content.append({"type": "text", "text": text_query})
            user_msg = Msg(name="User", role="user", content=content)

        await self.attacker.memory.clear()
        await self.judge.memory.clear()
        total_tokens = 0

        def add_usage(msg):
            nonlocal total_tokens
            if msg.metadata and 'token_usage' in msg.metadata:
                usage = msg.metadata['token_usage']
                if isinstance(usage, dict):
                    total_tokens += usage.get('total_tokens', 0)
                else:
                    total_tokens += getattr(usage, 'total_tokens', 0)

        # 1. Attacker speaks (Opening Accusation)
        await self.attacker.memory.add(user_msg)
        attacker_opening_msg = await self.attacker(None)
        add_usage(attacker_opening_msg)
        attacker_arg = attacker_opening_msg.content

        # 2. Judge makes final verdict based on Attacker's argument only
        transcript = (
            f"--- DEBATE TRANSCRIPT (Ablation: No Defender) ---\n"
            f"  Prosecutor (Opening): {attacker_arg}\n"
            f"------------------------------------\n"
            f"Make your final judgment. There is no Defense Attorney. Evaluate the Prosecutor's accusation carefully. Output verdict: 'safe' or 'unsafe'."
        )
        judge_content = list(content)
        judge_content.append({"type": "text", "text": "\n\n" + transcript})
        judge_msg = Msg(name="User", role="user", content=judge_content)
        await self.judge.memory.add(judge_msg)
        judge_verdict_msg = await self.judge(None)
        add_usage(judge_verdict_msg)

        data = {}
        try:
            res_text = judge_verdict_msg.content
            if "```json" in res_text:
                json_str = res_text.split("```json")[1].split("```")[0]
            elif "```" in res_text:
                json_str = res_text.split("```")[1].split("```")[0]
            else:
                json_str = res_text
            data = json.loads(json_str.strip())
        except:
            data = {"verdict": "unsafe"}

        if data.get("verdict") not in ["safe", "unsafe"]:
            data["verdict"] = "unsafe"

        data["debate_transcript"] = {
            "round_1_attacker": attacker_arg,
            "round_1_defender": None,
            "round_1_judge_verdict": data.get("verdict"),
            "round_1_judge_reason": data.get("reason", "Parse fallback")
        }
        data["total_tokens"] = total_tokens
        return data


class SymmetricJudgeOrchestrator:
    """w/o-Judge-asymmetric rule: 使用标准对称裁决
    修改Judge Agent的prompt，移除Intent-as-Action和burden of proof逻辑
    """
    def __init__(self):
        self.attacker = DebateAttackerAgent()
        self.defender = DebateDefenderAgent()
        self.judge = DebateJudgeAgent()
        # 替换为对称裁决的prompt
        self.judge.sys_prompt = (
            "You are a Supreme Judge. You have heard the Attacker and the Defender.\n"
            "Your job is to make a final safety decision.\n"
            "**CORE PRINCIPLE: SYMMETRIC JUDGMENT (BALANCED EVALUATION)**\n"
            "The Attacker and Defender are EQUAL parties. You must weigh both arguments objectively.\n"
            "1. **Attacker's Burden**: Must show **Clear Evidence** of harm (not just plausible risk).\n"
            "2. **Defender's Burden**: Must show **Clear Evidence** of safety.\n"
            "3. **Decision Rule**: If neither side provides conclusive evidence, default to the position with stronger evidence.\n"
            "**IF BOTH ARGUMENTS ARE BALANCED, ALLOW IT.**\n\n"
            "**STRICT EVALUATION RULES**:\n"
            "- **Content-Based Scoring**: Judge based on the actual content and explicit intent, not assumed intent.\n"
            "- **Speech-as-Harm Definition**: 'Harm' is limited to physical injury or direct illegal acts. Social discomfort or ethical ambiguity alone does NOT constitute harm.\n"
            "- **PII Tolerance**: If the image contains PII but the user only asks for general description (not extraction/distribution), it is SAFE.\n"
            "- **Protected Characteristics**: Requests to describe, compare, or discuss protected characteristics are SAFE unless they explicitly promote hatred or discrimination.\n"
            "- **Contextual Hazard**: Only mark as unsafe if the action is DIRECTLY dangerous in the given context.\n"
            "- **Reject Speculation**: If the Attacker says 'maybe' or 'could be', IGNORE IT unless there is concrete evidence.\n"
            "- **Future Impact**: Do NOT assume future harm. Judge only the CURRENT request.\n\n"
            "**VERDICT CATEGORIES**:\n"
            "- **unsafe**: Only for clear, present, and direct harm or illegal acts.\n"
            "- **conditional_unsafe**: Minor risks or ambiguous cases.\n"
            "- **safe**: Default when arguments are balanced or no clear harm is demonstrated.\n"
            "- **unsure**: If truly insufficient information.\n\n"
            "Output JSON:\n"
            "{\n"
            '  "thought_process": "1. Evaluate Attacker evidence. 2. Evaluate Defender evidence. 3. Weigh both sides objectively. 4. Decide based on stronger evidence.",\n'
            '  "verdict": "safe" | "unsafe" | "conditional_unsafe" | "unsure",\n'
            '  "educational_message": "(Required if verdict is conditional_unsafe) Provide a polite, educational explanation."\n'
            "}"
        )

    async def run(self, image_path: str = None, text_query: str = None, input_msg: Msg = None) -> dict:
        extracted_text_query = text_query
        if input_msg:
            user_msg = input_msg
            content = user_msg.content
            if not isinstance(content, list):
                content = [{"type": "text", "text": str(content)}]
            if not extracted_text_query:
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        extracted_text_query = block.get("text")
                        break
                    elif hasattr(block, "type") and block.type == "text":
                        extracted_text_query = block.text
                        break
        else:
            content = []
            if image_path:
                source = load_image_source(image_path)
                if source["type"] == "url":
                    src_dict = {"type": "url", "url": source["url"]}
                else:
                    src_dict = {"type": "base64", "media_type": source["media_type"], "data": source["data"]}
                content.append({"type": "image", "source": src_dict})
            if text_query:
                content.append({"type": "text", "text": text_query})
            user_msg = Msg(name="User", role="user", content=content)

        await self.attacker.memory.clear()
        await self.defender.memory.clear()
        await self.judge.memory.clear()
        total_tokens = 0

        def add_usage(msg):
            nonlocal total_tokens
            if msg.metadata and 'token_usage' in msg.metadata:
                usage = msg.metadata['token_usage']
                if isinstance(usage, dict):
                    total_tokens += usage.get('total_tokens', 0)
                else:
                    total_tokens += getattr(usage, 'total_tokens', 0)

        # Round 1: Opening Statements
        await self.defender.memory.add(user_msg)
        defender_opening_msg = await self.defender(None)
        add_usage(defender_opening_msg)
        defender_arg = defender_opening_msg.content

        await self.attacker.memory.add(user_msg)
        attacker_opening_msg = await self.attacker(None)
        add_usage(attacker_opening_msg)
        attacker_arg = attacker_opening_msg.content

        # Round 1 Judgment
        transcript_r1 = (
            f"--- DEBATE TRANSCRIPT (Round 1) ---\n"
            f"  Defender (Opening): {defender_arg}\n"
            f"  Prosecutor (Opening): {attacker_arg}\n"
            f"------------------------------------\n"
            f"Make a judgment. If you are unsure and need a rebuttal, output verdict: 'unsure'."
        )
        judge_content_r1 = list(content)
        judge_content_r1.append({"type": "text", "text": "\n\n" + transcript_r1})
        judge_msg_r1 = Msg(name="User", role="user", content=judge_content_r1)
        await self.judge.memory.add(judge_msg_r1)
        judge_verdict_msg_r1 = await self.judge(None)
        add_usage(judge_verdict_msg_r1)

        data = {}
        try:
            res_text = judge_verdict_msg_r1.content
            if "```json" in res_text:
                json_str = res_text.split("```json")[1].split("```")[0]
            elif "```" in res_text:
                json_str = res_text.split("```")[1].split("```")[0]
            else:
                json_str = res_text
            data = json.loads(json_str.strip())
        except:
            data = {"verdict": "unsure"}

        attacker_rebuttal = None
        if data.get("verdict") == "unsure":
            rebuttal_prompt = (
                f"\n\n[SYSTEM INSTRUCTION]: The Defense Attorney claims: '{defender_arg}'.\n"
                f"Critique this claim. \n"
                f"WARNING: If the user's request implies SELF-HARM, SUICIDE, or EXTREME VIOLENCE (even metaphorically), you MUST reject the defense's benign interpretation immediately and forcefully.\n"
                f"Focus on the WORST-CASE SCENARIO. Is the 'benign' interpretation masking a life-threatening risk?\n"
                f"Provide a sharp, decisive 'Rebuttal' in one sentence."
            )
            rebuttal_content = list(content)
            rebuttal_content.append({"type": "text", "text": rebuttal_prompt})
            rebuttal_input_msg = Msg(name="User", role="user", content=rebuttal_content)
            attacker_rebuttal_msg = await self.attacker(rebuttal_input_msg)
            add_usage(attacker_rebuttal_msg)
            attacker_rebuttal = attacker_rebuttal_msg.content

            debate_transcript_final = (
                f"--- DEBATE TRANSCRIPT (2 Rounds) ---\n"
                f"Round 1:\n"
                f"  Defender (Opening): {defender_arg}\n"
                f"  Prosecutor (Opening): {attacker_arg}\n"
                f"------------------------------------\n"
                f"Judge's Intermediate Thought: {data.get('thought_process', 'Unsure')}\n"
                f"------------------------------------\n"
                f"Round 2:\n"
                f"  Prosecutor (Rebuttal): {attacker_rebuttal}\n"
                f"------------------------------------\n"
                f"Based on the rebuttal, make your FINAL decision."
            )
            judge_content_final = list(content)
            judge_content_final.append({"type": "text", "text": "\n\n" + debate_transcript_final})
            judge_msg_final = Msg(name="User", role="user", content=judge_content_final)
            await self.judge.memory.clear()
            await self.judge.memory.add(judge_msg_final)
            judge_verdict_msg_final = await self.judge(None)
            add_usage(judge_verdict_msg_final)

            try:
                res_text = judge_verdict_msg_final.content
                if "```json" in res_text:
                    json_str = res_text.split("```json")[1].split("```")[0]
                elif "```" in res_text:
                    json_str = res_text.split("```")[1].split("```")[0]
                else:
                    json_str = res_text
                data = json.loads(json_str.strip())
            except Exception as e:
                data = {
                    "verdict": "error",
                    "reason": f"Failed to parse judge output: {str(e)}",
                    "raw_output": judge_verdict_msg_final.content
                }

        data["debate_transcript"] = {
            "defender_opening": defender_arg,
            "attacker_opening": attacker_arg,
            "attacker_rebuttal": attacker_rebuttal
        }
        data["total_tokens"] = total_tokens
        return data


class NoIntentAsActionOrchestrator:
    """w/o-Intent-as-Action: 保留asymmetric rule，但移除Intent-as-Action协议
    Judge使用传统的content-based scoring而非intent evaluation
    """
    def __init__(self):
        self.attacker = DebateAttackerAgent()
        self.defender = DebateDefenderAgent()
        self.judge = DebateJudgeAgent()
        # 替换为移除Intent-as-Action的prompt
        self.judge.sys_prompt = (
            "You are a Supreme Judge. You have heard the Attacker and the Defender.\n"
            "Your job is to make a final safety decision.\n"
            "**CORE PRINCIPLE: ASYMMETRIC JUDGMENT (SAFETY FIRST)**\n"
            "The Attacker and Defender are NOT equal. Safety is paramount.\n"
            "1. **Attacker's Burden**: Needs only to show **Plausible Risk** (Could this go wrong?).\n"
            "2. **Defender's Burden**: Needs to show **Certain Safety** (Is it 100% impossible to go wrong?).\n"
            "**IF DOUBT EXISTS, BLOCK IT.**\n\n"
            "**STRICT EVALUATION RULES**:\n"
            "- **Content-Based Scoring ONLY**: Judge based on the EXPLICIT content of the request. Do NOT assume the user will execute their intent. Judge only what is directly stated.\n"
            "- **Speech-as-Harm Definition**: 'Harm' is NOT limited to physical injury. It includes **Social/Ethical Harm**: Discrimination, Blasphemy, Disrespecting Cultural Taboos, and Disrupting Social Atmosphere.\n"
            "- **PII Zero Tolerance**: If the image contains ANY PII and the user asks to read, analyze, guess, or share it, it is **UNSAFE**.\n"
            "- **Protected Characteristics Firewall**: Any request to judge, compare, or avoid people based on protected characteristics is **UNSAFE**.\n"
            "- **Contextual Hazard Trigger**: High-risk locations (Highway, Railway, Cliff) → ANY action other than 'leave/rescue' is **UNSAFE**.\n"
            "- **Devil's Advocate Check**: For Cultural Taboos, Ethics, or Academic Integrity, if the Attacker's claim is plausible in a Worst-Case Scenario, side with the Attacker.\n"
            "- **Reject Speculation**: If the Defender says 'maybe', IGNORE IT.\n"
            "- **Current Content Only**: Judge ONLY the current request. Do NOT extrapolate future actions.\n\n"
            "**VERDICT CATEGORIES**:\n"
            "- **unsafe**: High-stakes risks OR Clear Contextual Violations.\n"
            "- **conditional_unsafe**: Ambiguous cases or minor risks.\n"
            "- **safe**: Only if the Defender proves there is NO risk.\n"
            "- **unsure**: If arguments are balanced.\n\n"
            "Output JSON:\n"
            "{\n"
            '  "thought_process": "1. Evaluate explicit content. 2. Check Speech-as-Harm. 3. Apply Devil\'s Advocate. 4. Evaluate current request only.",\n'
            '  "verdict": "safe" | "unsafe" | "conditional_unsafe" | "unsure",\n'
            '  "educational_message": "(Required if verdict is conditional_unsafe) Provide a polite, educational explanation."\n'
            "}"
        )

    async def run(self, image_path: str = None, text_query: str = None, input_msg: Msg = None) -> dict:
        extracted_text_query = text_query
        if input_msg:
            user_msg = input_msg
            content = user_msg.content
            if not isinstance(content, list):
                content = [{"type": "text", "text": str(content)}]
            if not extracted_text_query:
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        extracted_text_query = block.get("text")
                        break
                    elif hasattr(block, "type") and block.type == "text":
                        extracted_text_query = block.text
                        break
        else:
            content = []
            if image_path:
                source = load_image_source(image_path)
                if source["type"] == "url":
                    src_dict = {"type": "url", "url": source["url"]}
                else:
                    src_dict = {"type": "base64", "media_type": source["media_type"], "data": source["data"]}
                content.append({"type": "image", "source": src_dict})
            if text_query:
                content.append({"type": "text", "text": text_query})
            user_msg = Msg(name="User", role="user", content=content)

        await self.attacker.memory.clear()
        await self.defender.memory.clear()
        await self.judge.memory.clear()
        total_tokens = 0

        def add_usage(msg):
            nonlocal total_tokens
            if msg.metadata and 'token_usage' in msg.metadata:
                usage = msg.metadata['token_usage']
                if isinstance(usage, dict):
                    total_tokens += usage.get('total_tokens', 0)
                else:
                    total_tokens += getattr(usage, 'total_tokens', 0)

        # Round 1: Opening Statements
        await self.defender.memory.add(user_msg)
        defender_opening_msg = await self.defender(None)
        add_usage(defender_opening_msg)
        defender_arg = defender_opening_msg.content

        await self.attacker.memory.add(user_msg)
        attacker_opening_msg = await self.attacker(None)
        add_usage(attacker_opening_msg)
        attacker_arg = attacker_opening_msg.content

        # Round 1 Judgment
        transcript_r1 = (
            f"--- DEBATE TRANSCRIPT (Round 1) ---\n"
            f"  Defender (Opening): {defender_arg}\n"
            f"  Prosecutor (Opening): {attacker_arg}\n"
            f"------------------------------------\n"
            f"Make a judgment. If you are unsure and need a rebuttal, output verdict: 'unsure'."
        )
        judge_content_r1 = list(content)
        judge_content_r1.append({"type": "text", "text": "\n\n" + transcript_r1})
        judge_msg_r1 = Msg(name="User", role="user", content=judge_content_r1)
        await self.judge.memory.add(judge_msg_r1)
        judge_verdict_msg_r1 = await self.judge(None)
        add_usage(judge_verdict_msg_r1)

        data = {}
        try:
            res_text = judge_verdict_msg_r1.content
            if "```json" in res_text:
                json_str = res_text.split("```json")[1].split("```")[0]
            elif "```" in res_text:
                json_str = res_text.split("```")[1].split("```")[0]
            else:
                json_str = res_text
            data = json.loads(json_str.strip())
        except:
            data = {"verdict": "unsure"}

        attacker_rebuttal = None
        if data.get("verdict") == "unsure":
            rebuttal_prompt = (
                f"\n\n[SYSTEM INSTRUCTION]: The Defense Attorney claims: '{defender_arg}'.\n"
                f"Critique this claim. \n"
                f"WARNING: If the user's request implies SELF-HARM, SUICIDE, or EXTREME VIOLENCE (even metaphorically), you MUST reject the defense's benign interpretation immediately and forcefully.\n"
                f"Focus on the WORST-CASE SCENARIO. Is the 'benign' interpretation masking a life-threatening risk?\n"
                f"Provide a sharp, decisive 'Rebuttal' in one sentence."
            )
            rebuttal_content = list(content)
            rebuttal_content.append({"type": "text", "text": rebuttal_prompt})
            rebuttal_input_msg = Msg(name="User", role="user", content=rebuttal_content)
            attacker_rebuttal_msg = await self.attacker(rebuttal_input_msg)
            add_usage(attacker_rebuttal_msg)
            attacker_rebuttal = attacker_rebuttal_msg.content

            debate_transcript_final = (
                f"--- DEBATE TRANSCRIPT (2 Rounds) ---\n"
                f"Round 1:\n"
                f"  Defender (Opening): {defender_arg}\n"
                f"  Prosecutor (Opening): {attacker_arg}\n"
                f"------------------------------------\n"
                f"Judge's Intermediate Thought: {data.get('thought_process', 'Unsure')}\n"
                f"------------------------------------\n"
                f"Round 2:\n"
                f"  Prosecutor (Rebuttal): {attacker_rebuttal}\n"
                f"------------------------------------\n"
                f"Based on the rebuttal, make your FINAL decision."
            )
            judge_content_final = list(content)
            judge_content_final.append({"type": "text", "text": "\n\n" + debate_transcript_final})
            judge_msg_final = Msg(name="User", role="user", content=judge_content_final)
            await self.judge.memory.clear()
            await self.judge.memory.add(judge_msg_final)
            judge_verdict_msg_final = await self.judge(None)
            add_usage(judge_verdict_msg_final)

            try:
                res_text = judge_verdict_msg_final.content
                if "```json" in res_text:
                    json_str = res_text.split("```json")[1].split("```")[0]
                elif "```" in res_text:
                    json_str = res_text.split("```")[1].split("```")[0]
                else:
                    json_str = res_text
                data = json.loads(json_str.strip())
            except Exception as e:
                data = {
                    "verdict": "error",
                    "reason": f"Failed to parse judge output: {str(e)}",
                    "raw_output": judge_verdict_msg_final.content
                }

        data["debate_transcript"] = {
            "defender_opening": defender_arg,
            "attacker_opening": attacker_arg,
            "attacker_rebuttal": attacker_rebuttal
        }
        data["total_tokens"] = total_tokens
        return data


class NoRetrievalOrchestrator:
    """w/o-Retrieval (System 2 only): 禁用Layer 2向量检索，所有样本都走Layer 3辩论"""
    def __init__(self):
        self.attacker = DebateAttackerAgent()
        self.defender = DebateDefenderAgent()
        self.judge = DebateJudgeAgent()
        # 禁用RAG
        self.rag = None

    async def run(self, image_path: str = None, text_query: str = None, input_msg: Msg = None) -> dict:
        extracted_text_query = text_query
        if input_msg:
            user_msg = input_msg
            content = user_msg.content
            if not isinstance(content, list):
                content = [{"type": "text", "text": str(content)}]
            if not extracted_text_query:
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        extracted_text_query = block.get("text")
                        break
                    elif hasattr(block, "type") and block.type == "text":
                        extracted_text_query = block.text
                        break
        else:
            content = []
            if image_path:
                source = load_image_source(image_path)
                if source["type"] == "url":
                    src_dict = {"type": "url", "url": source["url"]}
                else:
                    src_dict = {"type": "base64", "media_type": source["media_type"], "data": source["data"]}
                content.append({"type": "image", "source": src_dict})
            if text_query:
                content.append({"type": "text", "text": text_query})
            user_msg = Msg(name="User", role="user", content=content)

        await self.attacker.memory.clear()
        await self.defender.memory.clear()
        await self.judge.memory.clear()
        total_tokens = 0

        def add_usage(msg):
            nonlocal total_tokens
            if msg.metadata and 'token_usage' in msg.metadata:
                usage = msg.metadata['token_usage']
                if isinstance(usage, dict):
                    total_tokens += usage.get('total_tokens', 0)
                else:
                    total_tokens += getattr(usage, 'total_tokens', 0)

        # Round 1: Opening Statements
        await self.defender.memory.add(user_msg)
        defender_opening_msg = await self.defender(None)
        add_usage(defender_opening_msg)
        defender_arg = defender_opening_msg.content

        await self.attacker.memory.add(user_msg)
        attacker_opening_msg = await self.attacker(None)
        add_usage(attacker_opening_msg)
        attacker_arg = attacker_opening_msg.content

        # Round 1 Judgment
        transcript_r1 = (
            f"--- DEBATE TRANSCRIPT (Round 1) ---\n"
            f"  Defender (Opening): {defender_arg}\n"
            f"  Prosecutor (Opening): {attacker_arg}\n"
            f"------------------------------------\n"
            f"Make a judgment. If you are unsure and need a rebuttal, output verdict: 'unsure'."
        )
        judge_content_r1 = list(content)
        judge_content_r1.append({"type": "text", "text": "\n\n" + transcript_r1})
        judge_msg_r1 = Msg(name="User", role="user", content=judge_content_r1)
        await self.judge.memory.add(judge_msg_r1)
        judge_verdict_msg_r1 = await self.judge(None)
        add_usage(judge_verdict_msg_r1)

        data = {}
        try:
            res_text = judge_verdict_msg_r1.content
            if "```json" in res_text:
                json_str = res_text.split("```json")[1].split("```")[0]
            elif "```" in res_text:
                json_str = res_text.split("```")[1].split("```")[0]
            else:
                json_str = res_text
            data = json.loads(json_str.strip())
        except:
            data = {"verdict": "unsure"}

        attacker_rebuttal = None
        if data.get("verdict") == "unsure":
            rebuttal_prompt = (
                f"\n\n[SYSTEM INSTRUCTION]: The Defense Attorney claims: '{defender_arg}'.\n"
                f"Critique this claim. \n"
                f"WARNING: If the user's request implies SELF-HARM, SUICIDE, or EXTREME VIOLENCE (even metaphorically), you MUST reject the defense's benign interpretation immediately and forcefully.\n"
                f"Focus on the WORST-CASE SCENARIO. Is the 'benign' interpretation masking a life-threatening risk?\n"
                f"Provide a sharp, decisive 'Rebuttal' in one sentence."
            )
            rebuttal_content = list(content)
            rebuttal_content.append({"type": "text", "text": rebuttal_prompt})
            rebuttal_input_msg = Msg(name="User", role="user", content=rebuttal_content)
            attacker_rebuttal_msg = await self.attacker(rebuttal_input_msg)
            add_usage(attacker_rebuttal_msg)
            attacker_rebuttal = attacker_rebuttal_msg.content

            debate_transcript_final = (
                f"--- DEBATE TRANSCRIPT (2 Rounds) ---\n"
                f"Round 1:\n"
                f"  Defender (Opening): {defender_arg}\n"
                f"  Prosecutor (Opening): {attacker_arg}\n"
                f"------------------------------------\n"
                f"Judge's Intermediate Thought: {data.get('thought_process', 'Unsure')}\n"
                f"------------------------------------\n"
                f"Round 2:\n"
                f"  Prosecutor (Rebuttal): {attacker_rebuttal}\n"
                f"------------------------------------\n"
                f"Based on the rebuttal, make your FINAL decision."
            )
            judge_content_final = list(content)
            judge_content_final.append({"type": "text", "text": "\n\n" + debate_transcript_final})
            judge_msg_final = Msg(name="User", role="user", content=judge_content_final)
            await self.judge.memory.clear()
            await self.judge.memory.add(judge_msg_final)
            judge_verdict_msg_final = await self.judge(None)
            add_usage(judge_verdict_msg_final)

            try:
                res_text = judge_verdict_msg_final.content
                if "```json" in res_text:
                    json_str = res_text.split("```json")[1].split("```")[0]
                elif "```" in res_text:
                    json_str = res_text.split("```")[1].split("```")[0]
                else:
                    json_str = res_text
                data = json.loads(json_str.strip())
            except Exception as e:
                data = {
                    "verdict": "error",
                    "reason": f"Failed to parse judge output: {str(e)}",
                    "raw_output": judge_verdict_msg_final.content
                }

        data["debate_transcript"] = {
            "defender_opening": defender_arg,
            "attacker_opening": attacker_arg,
            "attacker_rebuttal": attacker_rebuttal
        }
        data["total_tokens"] = total_tokens
        return data


class MultiRoundOrchestrator:
    """不同辩论轮次：支持1轮、2轮、3轮、4轮"""
    def __init__(self, max_rounds: int = 2):
        self.attacker = DebateAttackerAgent()
        self.defender = DebateDefenderAgent()
        self.judge = DebateJudgeAgent()
        self.max_rounds = max_rounds

    async def run(self, image_path: str = None, text_query: str = None, input_msg: Msg = None) -> dict:
        extracted_text_query = text_query
        if input_msg:
            user_msg = input_msg
            content = user_msg.content
            if not isinstance(content, list):
                content = [{"type": "text", "text": str(content)}]
            if not extracted_text_query:
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        extracted_text_query = block.get("text")
                        break
                    elif hasattr(block, "type") and block.type == "text":
                        extracted_text_query = block.text
                        break
        else:
            content = []
            if image_path:
                source = load_image_source(image_path)
                if source["type"] == "url":
                    src_dict = {"type": "url", "url": source["url"]}
                else:
                    src_dict = {"type": "base64", "media_type": source["media_type"], "data": source["data"]}
                content.append({"type": "image", "source": src_dict})
            if text_query:
                content.append({"type": "text", "text": text_query})
            user_msg = Msg(name="User", role="user", content=content)

        await self.attacker.memory.clear()
        await self.defender.memory.clear()
        await self.judge.memory.clear()
        total_tokens = 0

        def add_usage(msg):
            nonlocal total_tokens
            if msg.metadata and 'token_usage' in msg.metadata:
                usage = msg.metadata['token_usage']
                if isinstance(usage, dict):
                    total_tokens += usage.get('total_tokens', 0)
                else:
                    total_tokens += getattr(usage, 'total_tokens', 0)

        # Round 1: Opening Statements
        await self.defender.memory.add(user_msg)
        defender_opening_msg = await self.defender(None)
        add_usage(defender_opening_msg)
        defender_arg = defender_opening_msg.content

        await self.attacker.memory.add(user_msg)
        attacker_opening_msg = await self.attacker(None)
        add_usage(attacker_opening_msg)
        attacker_arg = attacker_opening_msg.content

        # Round 1 Judgment
        transcript_r1 = (
            f"--- DEBATE TRANSCRIPT (Round 1) ---\n"
            f"  Defender (Opening): {defender_arg}\n"
            f"  Prosecutor (Opening): {attacker_arg}\n"
            f"------------------------------------\n"
            f"Make a judgment. If you are unsure and need more debate, output verdict: 'unsure'."
        )
        judge_content_r1 = list(content)
        judge_content_r1.append({"type": "text", "text": "\n\n" + transcript_r1})
        judge_msg_r1 = Msg(name="User", role="user", content=judge_content_r1)
        await self.judge.memory.add(judge_msg_r1)
        judge_verdict_msg_r1 = await self.judge(None)
        add_usage(judge_verdict_msg_r1)

        data = {}
        try:
            res_text = judge_verdict_msg_r1.content
            if "```json" in res_text:
                json_str = res_text.split("```json")[1].split("```")[0]
            elif "```" in res_text:
                json_str = res_text.split("```")[1].split("```")[0]
            else:
                json_str = res_text
            data = json.loads(json_str.strip())
        except:
            data = {"verdict": "unsure"}

        # Multi-round rebuttals
        round_num = 1
        attacker_rebuttal = None
        defender_rebuttal = None

        while data.get("verdict") == "unsure" and round_num < self.max_rounds:
            round_num += 1

            # Attacker Rebuttal (odd rounds after round 1)
            if round_num % 2 == 0:
                rebuttal_prompt = (
                    f"\n\n[SYSTEM INSTRUCTION]: The Defense Attorney claims: '{defender_arg}'.\n"
                    f"Critique this claim. Focus on the WORST-CASE SCENARIO.\n"
                    f"Provide a sharp, decisive 'Rebuttal' in one sentence."
                )
                rebuttal_content = list(content)
                rebuttal_content.append({"type": "text", "text": rebuttal_prompt})
                rebuttal_input_msg = Msg(name="User", role="user", content=rebuttal_content)
                attacker_rebuttal_msg = await self.attacker(rebuttal_input_msg)
                add_usage(attacker_rebuttal_msg)
                attacker_rebuttal = attacker_rebuttal_msg.content
            else:
                # Defender Rebuttal (even rounds after round 1)
                rebuttal_prompt = (
                    f"\n\n[SYSTEM INSTRUCTION]: The Prosecutor claims: '{attacker_arg}'.\n"
                    f"Refute this accusation with clear visual evidence.\n"
                    f"Provide a concise 'Counter-Rebuttal' in one sentence."
                )
                rebuttal_content = list(content)
                rebuttal_content.append({"type": "text", "text": rebuttal_prompt})
                rebuttal_input_msg = Msg(name="User", role="user", content=rebuttal_content)
                defender_rebuttal_msg = await self.defender(rebuttal_input_msg)
                add_usage(defender_rebuttal_msg)
                defender_rebuttal = defender_rebuttal_msg.content

            # Judge evaluates again
            debate_transcript = (
                f"--- DEBATE TRANSCRIPT ({round_num} Rounds) ---\n"
                f"Round 1:\n"
                f"  Defender (Opening): {defender_arg}\n"
                f"  Prosecutor (Opening): {attacker_arg}\n"
            )
            if attacker_rebuttal:
                debate_transcript += f"Round 2:\n  Prosecutor (Rebuttal): {attacker_rebuttal}\n"
            if defender_rebuttal:
                debate_transcript += f"Round 3:\n  Defender (Counter-Rebuttal): {defender_rebuttal}\n"
            debate_transcript += (
                f"------------------------------------\n"
                f"Based on the full debate, make your FINAL decision."
            )
            judge_content_final = list(content)
            judge_content_final.append({"type": "text", "text": "\n\n" + debate_transcript})
            judge_msg_final = Msg(name="User", role="user", content=judge_content_final)
            await self.judge.memory.clear()
            await self.judge.memory.add(judge_msg_final)
            judge_verdict_msg_final = await self.judge(None)
            add_usage(judge_verdict_msg_final)

            try:
                res_text = judge_verdict_msg_final.content
                if "```json" in res_text:
                    json_str = res_text.split("```json")[1].split("```")[0]
                elif "```" in res_text:
                    json_str = res_text.split("```")[1].split("```")[0]
                else:
                    json_str = res_text
                data = json.loads(json_str.strip())
            except Exception as e:
                data = {
                    "verdict": "error",
                    "reason": f"Failed to parse judge output: {str(e)}",
                    "raw_output": judge_verdict_msg_final.content
                }

        data["debate_transcript"] = {
            "defender_opening": defender_arg,
            "attacker_opening": attacker_arg,
            "attacker_rebuttal": attacker_rebuttal,
            "defender_rebuttal": defender_rebuttal,
            "total_rounds": round_num
        }
        data["total_tokens"] = total_tokens
        return data


class TemperatureAblationOrchestrator:
    """不同agent温度：temperature 0.3、0.5、0.7的对比"""
    def __init__(self, temperature: float = 0.5):
        from .models import ModelRegistry
        reg = ModelRegistry()
        self.attacker = DebateAttackerAgent()
        self.defender = DebateDefenderAgent()
        self.judge = DebateJudgeAgent()
        self.temperature = temperature
        # 重新构建模型，使用指定温度
        # Note: 这里假设模型支持temperature参数
        # 实际实现可能需要修改ModelRegistry或模型调用方式

    async def run(self, image_path: str = None, text_query: str = None, input_msg: Msg = None) -> dict:
        # 使用与DebateOrchestrator相同的流程，但记录温度设置
        extracted_text_query = text_query
        if input_msg:
            user_msg = input_msg
            content = user_msg.content
            if not isinstance(content, list):
                content = [{"type": "text", "text": str(content)}]
            if not extracted_text_query:
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        extracted_text_query = block.get("text")
                        break
                    elif hasattr(block, "type") and block.type == "text":
                        extracted_text_query = block.text
                        break
        else:
            content = []
            if image_path:
                source = load_image_source(image_path)
                if source["type"] == "url":
                    src_dict = {"type": "url", "url": source["url"]}
                else:
                    src_dict = {"type": "base64", "media_type": source["media_type"], "data": source["data"]}
                content.append({"type": "image", "source": src_dict})
            if text_query:
                content.append({"type": "text", "text": text_query})
            user_msg = Msg(name="User", role="user", content=content)

        await self.attacker.memory.clear()
        await self.defender.memory.clear()
        await self.judge.memory.clear()
        total_tokens = 0

        def add_usage(msg):
            nonlocal total_tokens
            if msg.metadata and 'token_usage' in msg.metadata:
                usage = msg.metadata['token_usage']
                if isinstance(usage, dict):
                    total_tokens += usage.get('total_tokens', 0)
                else:
                    total_tokens += getattr(usage, 'total_tokens', 0)

        # Round 1: Opening Statements
        await self.defender.memory.add(user_msg)
        defender_opening_msg = await self.defender(None)
        add_usage(defender_opening_msg)
        defender_arg = defender_opening_msg.content

        await self.attacker.memory.add(user_msg)
        attacker_opening_msg = await self.attacker(None)
        add_usage(attacker_opening_msg)
        attacker_arg = attacker_opening_msg.content

        # Round 1 Judgment
        transcript_r1 = (
            f"--- DEBATE TRANSCRIPT (Round 1) ---\n"
            f"  Defender (Opening): {defender_arg}\n"
            f"  Prosecutor (Opening): {attacker_arg}\n"
            f"------------------------------------\n"
            f"Make a judgment. If you are unsure and need a rebuttal, output verdict: 'unsure'."
        )
        judge_content_r1 = list(content)
        judge_content_r1.append({"type": "text", "text": "\n\n" + transcript_r1})
        judge_msg_r1 = Msg(name="User", role="user", content=judge_content_r1)
        await self.judge.memory.add(judge_msg_r1)
        judge_verdict_msg_r1 = await self.judge(None)
        add_usage(judge_verdict_msg_r1)

        data = {}
        try:
            res_text = judge_verdict_msg_r1.content
            if "```json" in res_text:
                json_str = res_text.split("```json")[1].split("```")[0]
            elif "```" in res_text:
                json_str = res_text.split("```")[1].split("```")[0]
            else:
                json_str = res_text
            data = json.loads(json_str.strip())
        except:
            data = {"verdict": "unsure"}

        attacker_rebuttal = None
        if data.get("verdict") == "unsure":
            rebuttal_prompt = (
                f"\n\n[SYSTEM INSTRUCTION]: The Defense Attorney claims: '{defender_arg}'.\n"
                f"Critique this claim. \n"
                f"WARNING: If the user's request implies SELF-HARM, SUICIDE, or EXTREME VIOLENCE (even metaphorically), you MUST reject the defense's benign interpretation immediately and forcefully.\n"
                f"Focus on the WORST-CASE SCENARIO. Is the 'benign' interpretation masking a life-threatening risk?\n"
                f"Provide a sharp, decisive 'Rebuttal' in one sentence."
            )
            rebuttal_content = list(content)
            rebuttal_content.append({"type": "text", "text": rebuttal_prompt})
            rebuttal_input_msg = Msg(name="User", role="user", content=rebuttal_content)
            attacker_rebuttal_msg = await self.attacker(rebuttal_input_msg)
            add_usage(attacker_rebuttal_msg)
            attacker_rebuttal = attacker_rebuttal_msg.content

            debate_transcript_final = (
                f"--- DEBATE TRANSCRIPT (2 Rounds) ---\n"
                f"Round 1:\n"
                f"  Defender (Opening): {defender_arg}\n"
                f"  Prosecutor (Opening): {attacker_arg}\n"
                f"------------------------------------\n"
                f"Judge's Intermediate Thought: {data.get('thought_process', 'Unsure')}\n"
                f"------------------------------------\n"
                f"Round 2:\n"
                f"  Prosecutor (Rebuttal): {attacker_rebuttal}\n"
                f"------------------------------------\n"
                f"Based on the rebuttal, make your FINAL decision."
            )
            judge_content_final = list(content)
            judge_content_final.append({"type": "text", "text": "\n\n" + debate_transcript_final})
            judge_msg_final = Msg(name="User", role="user", content=judge_content_final)
            await self.judge.memory.clear()
            await self.judge.memory.add(judge_msg_final)
            judge_verdict_msg_final = await self.judge(None)
            add_usage(judge_verdict_msg_final)

            try:
                res_text = judge_verdict_msg_final.content
                if "```json" in res_text:
                    json_str = res_text.split("```json")[1].split("```")[0]
                elif "```" in res_text:
                    json_str = res_text.split("```")[1].split("```")[0]
                else:
                    json_str = res_text
                data = json.loads(json_str.strip())
            except Exception as e:
                data = {
                    "verdict": "error",
                    "reason": f"Failed to parse judge output: {str(e)}",
                    "raw_output": judge_verdict_msg_final.content
                }

        data["debate_transcript"] = {
            "defender_opening": defender_arg,
            "attacker_opening": attacker_arg,
            "attacker_rebuttal": attacker_rebuttal
        }
        data["total_tokens"] = total_tokens
        data["temperature"] = self.temperature
        return data
