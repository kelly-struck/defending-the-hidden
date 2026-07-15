import os
import json
import base64
from pathlib import Path
from agentscope.message import Msg, URLSource, Base64Source, TextBlock
from agentscope.pipeline import MsgHub
from .agents import DebateAttackerAgent, DebateDefenderAgent, DebateJudgeAgent
try:
    from .retrieval.rag_manager import RAGManager
except ImportError:
    RAGManager = None

def load_image_source(path_or_url: str):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return URLSource(type="url", url=path_or_url)
    p = Path(path_or_url)
    if not p.exists():
        # Fallback or error? Let's assume it might be a file:// url
        if path_or_url.startswith("file://"):
             return URLSource(type="url", url=path_or_url)
        raise FileNotFoundError(f"Image file not found: {path_or_url}")
        
    data = p.read_bytes()
    
    # Check size limit (approx 4MB to be safe for various APIs, though error said 10MB)
    # If > 4MB, resize
    if len(data) > 4 * 1024 * 1024:
        try:
            from PIL import Image
            import io
            with Image.open(io.BytesIO(data)) as img:
                img = img.convert("RGB")
                # Resize logic: reduce max dimension to 1024 or similar until it fits
                max_size = 1024
                while True:
                    img.thumbnail((max_size, max_size))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    new_data = buf.getvalue()
                    if len(new_data) < 4 * 1024 * 1024:
                        data = new_data
                        p = p.with_suffix(".jpg") # Treat as jpg now
                        break
                    max_size = int(max_size * 0.8)
                    if max_size < 100: # Safety break
                        break
        except Exception as e:
            print(f"Warning: Failed to resize large image {path_or_url}: {e}")

    b64 = base64.b64encode(data).decode("utf-8")
    # naive type guess
    mt = "image/png" if p.suffix.lower() in {".png"} else "image/jpeg"
    return Base64Source(type="base64", media_type=mt, data=b64)

class DebateOrchestrator:
    def __init__(self):
        self.attacker = DebateAttackerAgent()
        self.defender = DebateDefenderAgent()
        self.judge = DebateJudgeAgent()
        self.rag = RAGManager() if RAGManager else None
        
    async def run(self, image_path: str = None, text_query: str = None, input_msg: Msg = None) -> dict:
        """
        Run the dynamic 2-round debate flow:
        Round 1:
          - Defender: Argues for benign intent.
          - Attacker: Argues for malicious intent (Opening).
        Round 2:
          - Attacker: Rebuts the Defender's argument.
        Verdict:
          - Judge: Decides based on the full transcript.
        
        Args:
            image_path: Path/URL to image
            text_query: Text query
            input_msg: Optional pre-constructed Msg object (overrides image_path/text_query if provided)
        """
        
        # Prepare user message content
        # We need to extract text_query securely for the learning step later
        extracted_text_query = text_query
        
        if input_msg:
            user_msg = input_msg
            # Extract content for local usage if needed, or just use the msg
            content = user_msg.content
            # Ensure content is list for appending ops later
            if not isinstance(content, list):
                content = [{"type": "text", "text": str(content)}]
            
            # Extract text query from input_msg if we didn't get it explicitly
            if not extracted_text_query:
                # Try to find the first text block
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
        
        # Clear memories
        await self.attacker.memory.clear()
        await self.defender.memory.clear()
        await self.judge.memory.clear()
        
        total_tokens = 0
        
        def add_usage(msg):
            nonlocal total_tokens
            if msg.metadata and 'token_usage' in msg.metadata:
                usage = msg.metadata['token_usage']
                # usage might be dict or object
                if isinstance(usage, dict):
                    total_tokens += usage.get('total_tokens', 0)
                else:
                    # Try attribute access
                    total_tokens += getattr(usage, 'total_tokens', 0)

        # --- Round 1: Opening Statements ---
        
        # 1. Defender speaks (Benign Interpretation)
        # We give the user message to the defender
        await self.defender.memory.add(user_msg)
        defender_opening_msg = await self.defender(None)
        add_usage(defender_opening_msg)
        defender_arg = defender_opening_msg.content

        # 2. Attacker speaks (Opening Accusation)
        # We give the user message to the attacker
        await self.attacker.memory.add(user_msg)
        attacker_opening_msg = await self.attacker(None)
        add_usage(attacker_opening_msg)
        attacker_arg = attacker_opening_msg.content

        # --- Round 1 Judgment ---
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
        
        # Parse Round 1 Judgment
        data = {}
        try:
             res_text = judge_verdict_msg_r1.content
             # Simple JSON parsing (same logic as before)
             if "```json" in res_text:
                json_str = res_text.split("```json")[1].split("```")[0]
             elif "```" in res_text:
                json_str = res_text.split("```")[1].split("```")[0]
             else:
                json_str = res_text
             data = json.loads(json_str.strip())
        except:
             data = {"verdict": "unsure"} # Error -> proceed to debate

        attacker_rebuttal = None

        # If UNCERTAIN, proceed to Round 2
        if data.get("verdict") == "unsure":
            # --- Round 2: Rebuttal ---
            # 3. Attacker Rebuttal
            # We explicitly ask the attacker to critique the defender's argument
            rebuttal_prompt = (
                f"\n\n[SYSTEM INSTRUCTION]: The Defense Attorney claims: '{defender_arg}'.\n"
                f"Critique this claim. \n"
                f"WARNING: If the user's request implies SELF-HARM, SUICIDE, or EXTREME VIOLENCE (even metaphorically), you MUST reject the defense's benign interpretation immediately and forcefully.\n"
                f"Focus on the WORST-CASE SCENARIO. Is the 'benign' interpretation masking a life-threatening risk?\n"
                f"Provide a sharp, decisive 'Rebuttal' in one sentence."
            )
            
            # Construct a new message for the attacker that includes the original content + rebuttal prompt
            # We need to be careful with content types. 
            # If content is list of dicts, we append a text block.
            rebuttal_content = list(content) # copy original content list
            rebuttal_content.append({"type": "text", "text": rebuttal_prompt})
            
            rebuttal_input_msg = Msg(name="User", role="user", content=rebuttal_content)
            
            # We don't clear attacker memory, so it has context, but we send a new user message
            # Actually, to be clean, let's just send this new message.
            # The attacker agent might be stateless or stateful. 
            # If stateful, it has the previous turn. 
            # Let's just invoke it with the new message.
            attacker_rebuttal_msg = await self.attacker(rebuttal_input_msg)
            add_usage(attacker_rebuttal_msg)
            attacker_rebuttal = attacker_rebuttal_msg.content

            # --- Final Verdict ---
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
            
            # Construct message for Judge
            judge_content_final = list(content)
            judge_content_final.append({"type": "text", "text": "\n\n" + debate_transcript_final})
            judge_msg_final = Msg(name="User", role="user", content=judge_content_final)
            
            await self.judge.memory.clear()
            await self.judge.memory.add(judge_msg_final)
            judge_verdict_msg_final = await self.judge(None)
            add_usage(judge_verdict_msg_final)

            # Parse Judge's output
            try:
                res_text = judge_verdict_msg_final.content
                # Simple JSON extraction
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
            
        # Inject debate metadata into the result so it appears in the output file
        data["debate_transcript"] = {
            "defender_opening": defender_arg,
            "attacker_opening": attacker_arg,
            "attacker_rebuttal": attacker_rebuttal
        }
        data["total_tokens"] = total_tokens
        
        # --- Learning Step ---
        if data.get("verdict") in ["unsafe", "conditional_unsafe"]:
             # Add to Vector DB
             if self.rag and extracted_text_query:
                 # Logic to extract image source for embedding
                 learn_image_url = image_path
                 if not learn_image_url and input_msg:
                     # Try to hunt down an image source in input_msg
                     # This supports agentscope ImageBlock or dict based content
                     blocks = input_msg.content if isinstance(input_msg.content, list) else [input_msg.content]
                     for b in blocks:
                         # check for object with source
                         if hasattr(b, "source"):
                             # ImageBlock object
                             src = b.source
                             if hasattr(src, "url") and src.url:
                                 learn_image_url = src.url
                             elif hasattr(src, "data") and src.data:
                                 # Base64 data
                                 mt = getattr(src, "media_type", "image/jpeg")
                                 learn_image_url = f"data:{mt};base64,{src.data}"
                             elif isinstance(src, dict):
                                 if "url" in src: learn_image_url = src["url"]
                                 elif "data" in src: learn_image_url = f"data:{src.get('media_type', 'image/jpeg')};base64,{src['data']}"
                             break
                         elif isinstance(b, dict) and b.get("type") == "image":
                             # Dict block
                             src = b.get("source", {})
                             if "url" in src: learn_image_url = src["url"]
                             elif "data" in src: learn_image_url = f"data:{src.get('media_type', 'image/jpeg')};base64,{src['data']}"
                             break

                 # FIX: Store raw query + image (Multimodal) for high-fidelity retrieval
                 # Do NOT include reasoning in the embedding source.
                 self.rag.add_case(
                     text=extracted_text_query, 
                     image_url=learn_image_url,
                     metadata={
                         "verdict": data.get("verdict"),
                         "reason": str(data.get("thought_process", "")),
                         "timestamp": "experiment_time"
                     }
                 )
                 
        return data
