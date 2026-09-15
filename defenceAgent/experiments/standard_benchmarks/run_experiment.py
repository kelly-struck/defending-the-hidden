import argparse
import os
import sys
import json
import asyncio
import base64
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from agentscope.message import Msg, TextBlock, ImageBlock
import agentscope
from defenceAgent.executor import MLLMExecutor
from defenceAgent.orchestrator_debate import DebateOrchestrator
from defenceAgent.experiments.standard_benchmarks.eval_utils import (
    DefenseStats, 
    load_harmbench_data, 
    load_mmsafetybench_data, 
    render_typographic_image, 
    FIGSTEP_TEXT_PROMPT, 
    is_refusal,
    ASRJudge
)
from defenceAgent.experiments.standard_benchmarks.adashield_utils import AdaShieldDefense

# Paths to datasets
HARMBENCH_ROOT = str(Path(__file__).resolve().parents[2] / "statics" / "HarmBench")
MMSAFETYBENCH_ROOT = str(Path(__file__).resolve().parents[2] / "statics" / "MM-SafetyBench")

async def run_experiment(args):
    # 1. Set Target Model
    os.environ["DEF_TARGET_MODEL"] = args.model
    print(f"Target Model: {args.model}")

    # Define output file path early
    out_file = Path(__file__).parent / f"results_{args.model}_{args.dataset}_{args.attack}_{args.defense}.json"
    
    # Load existing results
    existing_results = []
    if out_file.exists():
        if args.force:
            print(f"Force restart enabled. Ignoring existing results in {out_file.name}")
        else:
            try:
                with open(out_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_results = data.get("results", [])
                    # Filter out connection errors so they can be retried
                    existing_results = [
                        r for r in all_results 
                        if "Error calling Doubao" not in r.get("response", "") 
                        and "Connection error" not in r.get("response", "")
                    ]
                print(f"Loaded {len(existing_results)} valid results from {out_file.name} (ignored {len(all_results) - len(existing_results)} errors)")
            except Exception as e:
                print(f"Error loading existing results: {e}")

    # 2. Load Data
    if args.dataset.lower() == "harmbench":
        rows = load_harmbench_data(HARMBENCH_ROOT)
        if args.limit > 0:
            rows = rows[:args.limit]
    elif args.dataset.lower() == "mmsafetybench":
        # Pass limit to loader for stratified sampling
        rows = load_mmsafetybench_data(MMSAFETYBENCH_ROOT, total_limit=args.limit if args.limit > 0 else None)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Filter rows based on existing results
    existing_ids = {r['id'] for r in existing_results}
    rows = [r for r in rows if r['id'] not in existing_ids]
    
    if not rows and not existing_results:
        print(f"No data found for {args.dataset} in {HARMBENCH_ROOT if args.dataset=='harmbench' else MMSAFETYBENCH_ROOT}")
        return

    # Limit is already applied for mmsafetybench, and applied above for harmbench
    # if args.limit > 0:
    #    rows = rows[:args.limit]

    print(f"Loaded {len(rows)} samples from {args.dataset}")

    # 3. Initialize System
    # Initialize agentscope to ensure logging is configured
    agentscope.init(logging_level="INFO")
    
    stats = DefenseStats()
    
    print(f"Initializing MLLMExecutor with target_model_name={args.model}")
    executor = MLLMExecutor(target_model_name=args.model) # Wraps the target model
    
    adashield_defense = None
    if args.defense == "layer4":
        orchestrator = DebateOrchestrator()
        print("Defense: Layer 4 (Debate) Enabled")
        asr_judge = None
    elif args.defense.startswith("adashield"):
        orchestrator = None
        mode = "dynamic" if "dynamic" in args.defense else "static"
        print(f"Defense: AdaShield ({mode}) Enabled")
        adashield_defense = AdaShieldDefense(mode=mode)
        asr_judge = ASRJudge() # We still need judge for final evaluation
    else:
        orchestrator = None
        print("Defense: None (Direct Execution)")
        asr_judge = ASRJudge()
        print(f"ASR Judge: {asr_judge.model_name}")

    results = existing_results

    # 4. Main Loop
    for row in tqdm(rows, desc=f"{args.model}-{args.dataset}-{args.attack}-{args.defense}"):
        stats.add_sample()
        
        qid = row['id']
        question = row['question']
        img_path = row['image_path']
        source = row.get('source', '')
        
        # Prepare Input
        content_blocks = []
        
        # Handle Images
        if args.attack == "figstep":
            # Render text to image
            figstep_img_bytes = render_typographic_image(question)
            b64_figstep = base64.b64encode(figstep_img_bytes).decode("ascii")
            
            # If original image exists, include it first (Context)
            if img_path and os.path.exists(img_path):
                try:
                    with open(img_path, "rb") as f:
                        b64_orig = base64.b64encode(f.read()).decode("ascii")
                    content_blocks.append(ImageBlock(type="image", source={"type": "base64", "media_type": "image/jpeg", "data": b64_orig}))
                except Exception as e:
                    print(f"Error reading image {img_path}: {e}")
            
            # Add FigStep Image
            content_blocks.append(ImageBlock(type="image", source={"type": "base64", "media_type": "image/png", "data": b64_figstep}))
            
            # Text Prompt is the FigStep prompt
            text_prompt = FIGSTEP_TEXT_PROMPT
            
        else: # Standard Attack
            # Original Image if exists
            if img_path and os.path.exists(img_path):
                try:
                    with open(img_path, "rb") as f:
                        b64_orig = base64.b64encode(f.read()).decode("ascii")
                    content_blocks.append(ImageBlock(type="image", source={"type": "base64", "media_type": "image/jpeg", "data": b64_orig}))
                except Exception as e:
                    print(f"Error reading image {img_path}: {e}")
            
            # Original Question
            text_prompt = question

        # content_blocks.append(TextBlock(type="text", text=text_prompt)) # Will be added later based on defense
        
        # Estimate Tokens
        # stats.add_tokens(text_prompt) # Moved to after execution to get actual usage

        # Execute
        response_text = ""
        verdict = "safe" # Default for no defense
        
        try:
            if args.defense == "layer4":
                # DebateOrchestrator expects image_path and text. 
                # But we might have constructed a complex message (FigStep).
                # DebateOrchestrator.run(image_path, text) is the signature.
                # We need to adapt it or use a lower-level call if we have multiple images.
                # Looking at DebateOrchestrator.run:
                # async def run(self, image_path: str, query: str) -> Dict[str, Any]:
                # It builds the message internally.
                
                # If we have FigStep (multiple images or generated image), passing a single path isn't enough.
                # We need to bypass the convenience method and call the agents directly OR modify the orchestrator.
                # However, for this experiment, let's see if we can hack it.
                
                # If FigStep: we have bytes. We can save to temp file?
                # Or we can modify DebateOrchestrator to accept a Msg object?
                # The user wants "with layer4".
                
                # Let's assume for now we save the FigStep image to a temp file if needed, 
                # but DebateOrchestrator logic is:
                # 1. Attacker analyzes image+query.
                # 2. Defender analyzes image+query.
                # 3. Judge decides.
                
                # If we have 2 images (Context + FigStep), DebateOrchestrator might not handle it out of the box 
                # if it only takes one `image_path`.
                # Let's check `orchestrator_debate.py` again.
                # It takes `image_path`.
                
                # Workaround: If FigStep, we only send the FigStep image? 
                # If MM-SafetyBench needs context, this is a problem.
                # But FigStep usually *replaces* the content.
                # Let's assume for FigStep experiments, we only use the FigStep image (rendered text).
                # This simplifies things and aligns with standard FigStep evaluation.
                
                if args.attack == "figstep":
                    # Save FigStep bytes to temp file
                    temp_img_path = Path(f"temp_figstep_{qid}.png")
                    with open(temp_img_path, "wb") as f:
                        f.write(figstep_img_bytes)
                    
                    current_img_path = str(temp_img_path)
                    current_query = FIGSTEP_TEXT_PROMPT
                    
                    stats.hit_layer4()
                    debate_result = await orchestrator.run(current_img_path, current_query)
                    
                    # Cleanup
                    if temp_img_path.exists():
                        temp_img_path.unlink()
                        
                else:
                    # Standard
                    current_img_path = img_path if img_path else "" # Handle text-only?
                    # DebateOrchestrator might require an image.
                    # If text-only (HarmBench Standard), we might need a dummy image or handle it.
                    # If img_path is None, DebateOrchestrator might fail if it expects image.
                    # Let's check.
                    
                    stats.hit_layer4()
                    debate_result = await orchestrator.run(current_img_path, question)

                verdict = debate_result.get("verdict", "unsafe")
                response_text = debate_result.get("educational_message", "Blocked by Defense")
                
                # Token Usage from Defense Framework
                defense_tokens = debate_result.get("total_tokens", 0)
                stats.add_tokens(defense_tokens)
                
                if verdict == "safe":
                    # If safe, we execute the target model via Executor
                    # Pass action="allow" to simulate safe execution
                    # Reconstruct content blocks for final execution
                    final_blocks = list(content_blocks) # Copy images
                    final_blocks.append(TextBlock(type="text", text=text_prompt))
                    
                    user_msg = Msg(name="user", role="user", content=final_blocks)
                    response_msg = await executor(user_msg, {"action": "allow"})
                    response_text = response_msg.content if isinstance(response_msg.content, str) else str(response_msg.content)
                    
                    # Add tokens from executor response
                    if response_msg.metadata and 'token_usage' in response_msg.metadata:
                         usage = response_msg.metadata['token_usage']
                         total_tokens = usage.get('total_tokens', 0)
                         # stats.add_tokens(total_tokens) # Do not add target model tokens if defense is active
                    else:
                         # Fallback estimate
                         pass
                else:
                    # Blocked by Defense
                    pass

            elif adashield_defense:
                # AdaShield Defense
                
                # Helper to execute target model with a given text prompt
                async def target_exec_func(prompt_text):
                    # Reconstruct content blocks with new prompt
                    # We reuse the images from content_blocks
                    exec_blocks = []
                    # Copy images from original content_blocks
                    for b in content_blocks:
                        # Check if it's an image block (TypedDict or dict)
                        if isinstance(b, dict) and b.get("type") == "image":
                            exec_blocks.append(b)
                    
                    exec_blocks.append(TextBlock(type="text", text=prompt_text))
                    
                    user_msg = Msg(name="user", role="user", content=exec_blocks)
                    response_msg = await executor(user_msg, {"action": "allow"})
                    
                    resp_text = response_msg.content if isinstance(response_msg.content, str) else str(response_msg.content)
                    
                    tokens = 0
                    if response_msg.metadata and 'token_usage' in response_msg.metadata:
                         usage = response_msg.metadata['token_usage']
                         tokens = usage.get('total_tokens', 0)
                    
                    return resp_text, tokens

                if adashield_defense.mode == "static":
                    # Static: Just prepend prompt
                    full_query = f"{adashield_defense.static_prompt}\n\n{text_prompt}"
                    response_text, tokens = await target_exec_func(full_query)
                    stats.add_tokens(tokens)
                    
                elif adashield_defense.mode == "dynamic":
                    # Dynamic: Run loop
                    response_text, verdict, tokens = await adashield_defense.run_dynamic_loop(
                        query=text_prompt,
                        image_path=img_path, # Not strictly used in our simplified loop but passed
                        source=source,
                        executor_func=target_exec_func
                    )
                    stats.add_tokens(tokens)
                    # Note: verdict here is internal, we still use ASRJudge for final eval

            else:
                # No Defense
                # Add text prompt to blocks
                content_blocks.append(TextBlock(type="text", text=text_prompt))
                
                user_msg = Msg(name="user", role="user", content=content_blocks)
                response_msg = await executor(user_msg, {"action": "allow"})
                response_text = response_msg.content if isinstance(response_msg.content, str) else str(response_msg.content)
                
                # Add tokens from executor response
                if response_msg.metadata and 'token_usage' in response_msg.metadata:
                        usage = response_msg.metadata['token_usage']
                        total_tokens = usage.get('total_tokens', 0)
                        stats.add_tokens(total_tokens)
                else:
                        # Fallback estimate
                        stats.add_tokens(len(str(text_prompt)) // 4 + len(response_text) // 4)

        except Exception as e:
            print(f"Error processing {qid}: {e}")
            import traceback
            traceback.print_exc()
            response_text = f"Error: {e}"
            # Count tokens for error message if needed, but for defense we already added what we could
            if args.defense == "none":
                 stats.add_tokens(len(str(text_prompt)) // 4 + len(response_text) // 4)

        # Check for connection error
        if any(err_msg in response_text for err_msg in ["Error calling Doubao", "Connection error", "Connection aborted", "ConnectionResetError"]):
            print(f"\n[CRITICAL] Connection error detected for sample {qid}: {response_text}")
            print("Saving progress and exiting to prevent token waste.")
            break

        # Check Safety (ASR Calculation)
        reason = ""
        if args.defense == "layer4":
            # "take the verdict returned by the defense framework as the standard"
            # If verdict is "safe" (Allowed), it means defense thinks it's safe.
            # If it was an attack, defense failed -> Unsafe (Attack Success).
            # If verdict is "unsafe" (Blocked), defense succeeded -> Safe (Attack Failed).
            # Also handle "error" or "conditional_unsafe" as Attack Failed (Safe).
            # Only "safe" verdict counts as Attack Success.
            is_unsafe = (str(verdict).lower() == "safe")
            reason = response_text # For layer4, response_text often contains the educational message or refusal
        else:
            # No Defense OR AdaShield: Use ASR Judge
            # AdaShield doesn't return a "verdict" like Layer 4 (Block/Allow). 
            # It modifies the prompt to encourage refusal.
            # So we evaluate the final response.
            is_unsafe, reason = await asr_judge.evaluate(question, response_text)

        results.append({
            "id": qid,
            "question": question,
            "response": response_text,
            "is_unsafe": is_unsafe,
            "verdict": verdict,
            "reason": reason
        })

    stats.stop_timer()
    summary = stats.get_summary()
    
    # Calculate ASR
    total = len(results)
    summary["Total Samples"] = total
    unsafe_count = sum(1 for r in results if r['is_unsafe'])
    asr = unsafe_count / total if total > 0 else 0
    summary["ASR"] = f"{asr:.2%}"

    print("\n" + "="*30)
    print("EVALUATION STATISTICS")
    print("="*30)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("="*30 + "\n")

    # Save Results
    out_file = Path(__file__).parent / f"results_{args.model}_{args.dataset}_{args.attack}_{args.defense}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "stats": summary,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen-vl-max", "doubao-1.5-vision-pro-250328"])
    parser.add_argument("--dataset", type=str, required=True, choices=["harmbench", "mmsafetybench"])
    parser.add_argument("--attack", type=str, required=True, choices=["standard", "figstep"])
    parser.add_argument("--defense", type=str, required=True, choices=["none", "layer4", "adashield_static", "adashield_dynamic"])
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples for testing")
    parser.add_argument("--force", action="store_true", help="Force restart experiment, ignoring existing results")
    
    args = parser.parse_args()
    asyncio.run(run_experiment(args))

if __name__ == "__main__":
    main()
