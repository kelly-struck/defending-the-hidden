"""
Enhanced Ablation Study Runner — Stages 8-9 (Debate Rounds + Temperature)

Key features:
  1. DataInspectionFailed → treated as "blocked" (defense success)
  2. Per-sample detailed output saved to JSONL for manual review
  3. Checkpoint support — resume on interruption
  4. Insufficient balance detection → save checkpoint and exit
  5. Saves defense framework output (verdict, reasoning, debate transcript)
"""
from __future__ import annotations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root

import asyncio
import json
import time
import sys
import os
import base64
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / "defenceAgent" / ".env")

from agentscope.message import Msg, ImageBlock, TextBlock, Base64Source

from defenceAgent.orchestrator_ablation import (
    MultiRoundOrchestrator,
    TemperatureAblationOrchestrator,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIUO_IMAGE_DIR = PROJECT_ROOT / "defenceAgent" / "statics" / "SIUO" / "images"
SIUO_GEN_JSON = PROJECT_ROOT / "defenceAgent" / "statics" / "SIUO" / "siuo_gen.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "ablation_full"

CONTENT_MOD_ERRORS = [
    "DataInspectionFailed",
    "inappropriate content",
    "Input data may contain inappropriate",
]

BALANCE_ERRORS = [
    "insufficient",
    "Arrearage",
    "balance",
    "quota exceeded",
    "not enough",
    "欠费",
    "余额不足",
]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_siuo_gen() -> List[Dict[str, Any]]:
    with open(SIUO_GEN_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def build_user_msg(image_name: str, text: str) -> Msg:
    blocks = []
    if text:
        blocks.append(TextBlock(type="text", text=text))
    if image_name:
        p = SIUO_IMAGE_DIR / image_name
        if p.is_file():
            data = p.read_bytes()
            b64 = base64.b64encode(data).decode("utf-8")
            mt = "image/png" if p.suffix.lower() in {".png"} else "image/jpeg"
            src = Base64Source(type="base64", media_type=mt, data=b64)
            blocks.append(ImageBlock(type="image", source=src))
    return Msg(name="user", role="user", content=blocks)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def is_content_mod_error(error_str: str) -> bool:
    return any(kw in error_str for kw in CONTENT_MOD_ERRORS)


def is_balance_error(error_str: str) -> bool:
    return any(kw.lower() in error_str.lower() for kw in BALANCE_ERRORS)


def classify_verdict(result: dict, error_str: Optional[str] = None) -> str:
    if error_str:
        if is_content_mod_error(error_str):
            return "blocked"
        if is_balance_error(error_str):
            return "balance_error"
        return "error"

    verdict = result.get("verdict", "error")
    if verdict in ("unsafe", "conditional_unsafe"):
        return "blocked"
    elif verdict == "safe":
        return "allowed"
    else:
        return "error"


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: Path) -> Dict:
    if ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_stages": [], "stage_progress": {}}


def save_checkpoint(ckpt_path: Path, data: Dict):
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Orchestrator runner
# ---------------------------------------------------------------------------

async def run_single_sample(orchestrator, user_msg: Msg, image_name: str, text: str) -> dict:
    t0 = time.perf_counter()
    record = {
        "image": image_name,
        "text": text,
        "verdict": None,
        "reason": None,
        "thought_process": None,
        "total_tokens": 0,
        "elapsed_sec": 0,
        "error": None,
        "is_content_mod": False,
        "debate_transcript": None,
    }

    try:
        result = await orchestrator.run(input_msg=user_msg)
        record["verdict"] = result.get("verdict", "error")
        record["reason"] = result.get("reason", result.get("educational_message"))
        record["thought_process"] = result.get("thought_process")
        record["total_tokens"] = result.get("total_tokens", 0)
        record["debate_transcript"] = result.get("debate_transcript")
    except Exception as e:
        error_str = str(e)
        record["error"] = error_str
        record["is_content_mod"] = is_content_mod_error(error_str)
        if is_content_mod_error(error_str):
            record["verdict"] = "blocked_by_api"
            record["reason"] = "DashScope content moderation (treated as defense success)"
        elif is_balance_error(error_str):
            record["verdict"] = "balance_error"
        else:
            record["verdict"] = "error"

    record["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    return record


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

async def run_stage(
    stage_key: str,
    stage_name: str,
    orchestrator,
    dataset: List[Dict],
    output_dir: Path,
    checkpoint: Dict,
) -> Dict[str, Any]:
    if stage_key in checkpoint.get("completed_stages", []):
        print(f"  [SKIP] {stage_name} — already completed (checkpoint)")
        results_path = output_dir / f"{stage_key}_results.json"
        if results_path.exists():
            with open(results_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    jsonl_path = output_dir / f"{stage_key}_details.jsonl"
    results_path = output_dir / f"{stage_key}_results.json"
    ckpt_path = output_dir / "checkpoint.json"

    stage_progress = checkpoint.get("stage_progress", {}).get(stage_key, {})
    completed_indices = set(stage_progress.get("completed_indices", []))

    if completed_indices and jsonl_path.exists():
        print(f"  [RESUME] {stage_name} — {len(completed_indices)} samples already done")

    print(f"\n{'='*60}")
    print(f"  Stage: {stage_name} ({stage_key})")
    print(f"{'='*60}")

    blocked = 0
    allowed = 0
    errors = 0
    total_tokens = 0
    total_time = 0.0
    balance_exit = False

    for i, item in enumerate(dataset):
        if i in completed_indices:
            continue

        image_name = item.get("image", "")
        text = item.get("question", "")
        user_msg = build_user_msg(image_name, text)

        record = await run_single_sample(orchestrator, user_msg, image_name, text)
        record["sample_index"] = i
        record["question_id"] = item.get("question_id")
        record["category"] = item.get("category", "unknown")

        final = classify_verdict(record, record.get("error"))
        if final == "blocked":
            blocked += 1
        elif final == "allowed":
            allowed += 1
        elif final == "balance_error":
            print(f"\n  [ABORT] Insufficient API balance at sample {i}!")
            balance_exit = True
            break
        else:
            errors += 1

        total_tokens += record.get("total_tokens", 0)
        total_time += record.get("elapsed_sec", 0)

        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        completed_indices.add(i)

        done = len(completed_indices)
        if done % 10 == 0 or done == len(dataset):
            print(f"  [{stage_key}] {done}/{len(dataset)} | blocked={blocked} allowed={allowed} errors={errors}")

        checkpoint.setdefault("stage_progress", {}).setdefault(stage_key, {})
        checkpoint["stage_progress"][stage_key]["completed_indices"] = sorted(completed_indices)
        save_checkpoint(ckpt_path, checkpoint)

    n = len(completed_indices)
    summary = {
        "experiment_name": stage_name,
        "stage_key": stage_key,
        "total_samples": n,
        "blocked": blocked,
        "allowed": allowed,
        "errors": errors,
        "dsr": blocked / n if n > 0 else 0,
        "asr": allowed / n if n > 0 else 0,
        "avg_tokens": total_tokens / n if n > 0 else 0,
        "avg_time_sec": total_time / n if n > 0 else 0,
        "balance_exit": balance_exit,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not balance_exit:
        checkpoint.setdefault("completed_stages", []).append(stage_key)
        save_checkpoint(ckpt_path, checkpoint)

    print(f"\n  Result: DSR={summary['dsr']:.3f} ASR={summary['asr']:.3f} "
          f"Tokens={summary['avg_tokens']:.0f} Time={summary['avg_time_sec']:.1f}s")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.json"

    print("=" * 60)
    print("  Enhanced Ablation Study — Stages 8-9")
    print("  Stage 8: Debate Rounds (1/2/3/4)")
    print("  Stage 9: Temperature (0.3/0.5/0.7)")
    print("=" * 60)
    print(f"Output directory: {output_dir}")

    dataset = load_siuo_gen()
    print(f"Dataset: {len(dataset)} samples")

    checkpoint = load_checkpoint(ckpt_path)
    if checkpoint.get("completed_stages"):
        print(f"Existing checkpoint has: {checkpoint['completed_stages']}")

    # Stage 8: Debate Rounds
    stages = []
    for rounds in [1, 2, 3, 4]:
        stages.append((
            f"rounds_{rounds}",
            f"{rounds}-Round Debate",
            MultiRoundOrchestrator(max_rounds=rounds),
        ))

    # Stage 9: Temperature
    for temp in [0.3, 0.5, 0.7]:
        stages.append((
            f"temp_{temp}",
            f"Temperature {temp}",
            TemperatureAblationOrchestrator(temperature=temp),
        ))

    all_results = {}
    balance_exit = False

    for stage_key, stage_name, orchestrator in stages:
        try:
            result = await run_stage(
                stage_key, stage_name, orchestrator,
                dataset, output_dir, checkpoint
            )
            all_results[stage_key] = result
            if result.get("balance_exit"):
                balance_exit = True
                break
        except Exception as e:
            print(f"\n[FATAL] Stage {stage_name} crashed: {e}")
            traceback.print_exc()
            save_checkpoint(ckpt_path, checkpoint)
            break

    # Final summary
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY (Stages 8-9)")
    print("=" * 60)

    for key, r in all_results.items():
        status = "DONE" if key in checkpoint.get("completed_stages", []) else "PARTIAL"
        print(f"  [{status}] {r['experiment_name']}: "
              f"DSR={r['dsr']:.3f} ASR={r['asr']:.3f} "
              f"blocked={r['blocked']}/{r['total_samples']} "
              f"Tokens={r['avg_tokens']:.0f} Time={r['avg_time_sec']:.1f}s")

    if balance_exit:
        print("\n  [EXIT] Stopped due to insufficient API balance.")
    else:
        print("\n  All stages completed.")

    # Save combined summary for stages 8-9
    summary_path = output_dir / "summary_stages89.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
