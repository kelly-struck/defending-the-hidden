"""Real FPR Evaluation on the Benign Dataset.

Runs the actual DefenceOrchestrator (4-layer cascade: L1 cache → L2 vector DB
→ L4 debate) on benign samples to measure the False Positive Rate (FPR).

A false positive occurs when the defense system incorrectly blocks a benign
input, i.e., a safe query is misclassified as unsafe.

Usage:
    python evaluate_fpr_real.py --limit 10 --output-dir results/fpr_real
    python evaluate_fpr_real.py --output-dir results/fpr_real
    python evaluate_fpr_real.py --output-dir results/fpr_real --resume
"""
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root

import asyncio
import json
import os
import sys
import time
import argparse
import base64
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict

# AgentScope imports
from agentscope.message import Msg, ImageBlock, TextBlock, Base64Source

# Project imports
from defenceAgent.orchestrator import DefenceOrchestrator
from defenceAgent.models import ModelRegistry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_PATH = r"d:\AgentScope\datasets\benign\benign_dataset.json"
DEFAULT_OUTPUT_DIR = r"d:\AgentScope\results\fpr_real"

BALANCE_ERRORS = [
    "insufficient",
    "Arrearage",
    "balance",
    "quota exceeded",
    "not enough",
    "欠费",
    "余额不足",
]

CONTENT_MOD_ERRORS = [
    "DataInspectionFailed",
    "inappropriate content",
    "Input data may contain inappropriate",
]


def is_balance_error(error_str: str) -> bool:
    return any(kw.lower() in error_str.lower() for kw in BALANCE_ERRORS)


def is_content_mod_error(error_str: str) -> bool:
    return any(kw.lower() in error_str.lower() for kw in CONTENT_MOD_ERRORS)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_benign_dataset(path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load the benign dataset from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    # Filter out samples with null images (can't be evaluated by multimodal pipeline)
    valid = [s for s in samples if s.get('image') is not None]
    skipped = len(samples) - len(valid)
    if skipped > 0:
        print(f"Skipped {skipped} samples with null images")
    
    if limit:
        valid = valid[:limit]
    
    return valid


def build_user_msg(sample: Dict) -> Optional[Msg]:
    """Build a Msg object from a benign dataset sample.
    
    Handles base64-encoded images from MME/MM-Vet parquet files.
    Returns None if the message cannot be constructed.
    """
    question = sample.get('question', '')
    image_data = sample.get('image')
    
    blocks = [TextBlock(type="text", text=question)]
    
    if image_data is not None:
        # Image can be a dict with base64 data, or a raw bytes-like object
        if isinstance(image_data, dict):
            if image_data.get('type') == 'base64':
                b64_data = image_data.get('data', '')
                media_type = image_data.get('media_type', 'image/png')
                blocks.append(ImageBlock(
                    type="image",
                    source=Base64Source(
                        type="base64",
                        media_type=media_type,
                        data=b64_data,
                    )
                ))
            elif 'bytes' in image_data:
                bytes_field = image_data['bytes']
                if isinstance(bytes_field, dict):
                    # Nested format: bytes = {type, data, media_type}
                    b64_data = bytes_field.get('data', '')
                    media_type = bytes_field.get('media_type', 'image/png')
                    blocks.append(ImageBlock(
                        type="image",
                        source=Base64Source(
                            type="base64",
                            media_type=media_type,
                            data=b64_data,
                        )
                    ))
                else:
                    # Parquet format: raw bytes
                    b64_data = base64.b64encode(bytes_field).decode('ascii')
                    blocks.append(ImageBlock(
                        type="image",
                        source=Base64Source(
                            type="base64",
                            media_type="image/jpeg",
                            data=b64_data,
                        )
                    ))
            elif 'path' in image_data:
                # Image path format
                img_path = image_data['path']
                if os.path.exists(img_path):
                    with open(img_path, 'rb') as f:
                        b64_data = base64.b64encode(f.read()).decode('ascii')
                    blocks.append(ImageBlock(
                        type="image",
                        source=Base64Source(
                            type="base64",
                            media_type="image/jpeg",
                            data=b64_data,
                        )
                    ))
        elif isinstance(image_data, str) and len(image_data) > 200:
            # Likely a raw base64 string
            blocks.append(ImageBlock(
                type="image",
                source=Base64Source(
                    type="base64",
                    media_type="image/jpeg",
                    data=image_data,
                )
            ))
    
    return Msg(name="user", role="user", content=blocks)


# ---------------------------------------------------------------------------
# Defense evaluation
# ---------------------------------------------------------------------------

class BalanceError(Exception):
    """Raised when API balance is insufficient."""
    pass


def extract_action_and_source(
    safety_result: Any
) -> Tuple[str, str, Dict]:
    """Extract the defense action, source layer, and metadata from a SafetyResult.
    
    Returns: (action, source, metadata)
    - action: "allow" | "block" | "revise"
    - source: "layer1_cache" | "vector_db" | "layer4_debate" | "unknown"
    - metadata: full metadata dict
    """
    if safety_result is None:
        return "allow", "unknown", {}
    
    try:
        meta = {}
        if hasattr(safety_result, 'raw') and safety_result.raw is not None:
            if hasattr(safety_result.raw, 'metadata') and safety_result.raw.metadata:
                meta = safety_result.raw.metadata
        
        # Also check the SafetyResult fields directly
        if hasattr(safety_result, 'source'):
            source = safety_result.source
        else:
            source = meta.get('source', 'unknown')
        
        # Determine action
        action = meta.get('action', None)
        if action is None:
            # Fallback: check the safe field
            if hasattr(safety_result, 'safe'):
                action = "allow" if safety_result.safe else "block"
            else:
                action = "allow"
        
        # Normalize action
        action = str(action).lower()
        if action not in ("allow", "block", "revise"):
            action = "block"
        
        return action, source, meta
        
    except Exception as e:
        return "allow", "error", {"error": str(e)}


async def evaluate_single_sample(
    orchestrator: DefenceOrchestrator,
    sample: Dict,
) -> Dict[str, Any]:
    """Evaluate one benign sample through the real defense pipeline.
    
    Returns a result dict with action, source, timing, and metadata.
    """
    sample_id = sample.get('id', 'unknown')
    
    # Build message
    user_msg = build_user_msg(sample)
    if user_msg is None:
        return {
            "id": sample_id,
            "domain": sample.get("domain", "unknown"),
            "modality": sample.get("modality", "unknown"),
            "source_origin": sample.get("source", "unknown"),
            "question": sample.get("question", ""),
            "action": "error",
            "defense_source": "error",
            "is_false_positive": False,
            "elapsed_sec": 0,
            "error": "failed to build message",
        }
    
    # Run through defense pipeline
    start_time = time.time()
    try:
        final_result, _, _ = await orchestrator.run(user_msg)
        elapsed = time.time() - start_time
        
        action, source, meta = extract_action_and_source(final_result)
        
        is_fp = (action == "block")  # Benign sample blocked = false positive
        
        return {
            "id": sample_id,
            "domain": sample.get("domain", "unknown"),
            "modality": sample.get("modality", "unknown"),
            "source_origin": sample.get("source", "unknown"),
            "question": sample.get("question", "")[:200],
            "action": action,
            "defense_source": source,
            "is_false_positive": is_fp,
            "elapsed_sec": round(elapsed, 2),
            "metadata_summary": {
                "verdict": meta.get("verdict", "unknown"),
                "score": meta.get("score", 0),
                "confidence": meta.get("confidence", 0),
                "total_tokens": meta.get("total_tokens", 0),
            },
        }
    
    except Exception as e:
        elapsed = time.time() - start_time
        error_str = str(e)
        
        if is_balance_error(error_str):
            raise BalanceError(error_str)
        
        if is_content_mod_error(error_str):
            return {
                "id": sample_id,
                "domain": sample.get("domain", "unknown"),
                "modality": sample.get("modality", "unknown"),
                "source_origin": sample.get("source", "unknown"),
                "question": sample.get("question", "")[:200],
                "action": "block",
                "defense_source": "api_content_moderation",
                "is_false_positive": True,
                "elapsed_sec": round(elapsed, 2),
                "error": "DataInspectionFailed",
            }
        
        return {
            "id": sample_id,
            "domain": sample.get("domain", "unknown"),
            "modality": sample.get("modality", "unknown"),
            "source_origin": sample.get("source", "unknown"),
            "question": sample.get("question", "")[:200],
            "action": "error",
            "defense_source": "error",
            "is_false_positive": False,
            "elapsed_sec": round(elapsed, 2),
            "error": error_str,
        }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def run_evaluation(
    samples: List[Dict],
    output_dir: Path,
    resume: bool = True,
    batch_save_interval: int = 10,
) -> List[Dict]:
    """Run FPR evaluation on all samples."""
    
    output_file = output_dir / "fpr_raw_results.json"
    
    # Resume logic
    processed_ids = set()
    results = []
    if resume and output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        if isinstance(existing, dict) and 'details' in existing:
            results = existing['details']
            processed_ids = {r['id'] for r in results}
            print(f"Resumed: {len(processed_ids)} samples already evaluated")
    
    # Initialize orchestrator
    print("Initializing DefenceOrchestrator...")
    orchestrator = DefenceOrchestrator()
    
    total = len(samples)
    remaining = [s for s in samples if s.get('id') not in processed_ids]
    print(f"Remaining to evaluate: {len(remaining)} / {total}")
    
    # Evaluate
    balance_exit = False
    for i, sample in enumerate(remaining):
        sid = sample.get('id', f'unknown_{i}')
        print(f"\n[{len(results)+1}/{total}] ID={sid}, "
              f"domain={sample.get('domain', '?')}, "
              f"modality={sample.get('modality', '?')}")
        
        try:
            result = await evaluate_single_sample(orchestrator, sample)
        except BalanceError as e:
            print(f"\n[ABORT] Insufficient API balance: {e}")
            balance_exit = True
            break
        
        results.append(result)
        
        # Print result
        fp_marker = " [FP!]" if result['is_false_positive'] else ""
        err = result.get('error', '')
        if err:
            print(f"  → ERROR: {err} ({result['elapsed_sec']:.1f}s)")
        else:
            print(f"  → {result['action']} "
                  f"(layer: {result['defense_source']}, "
                  f"{result['elapsed_sec']:.1f}s){fp_marker}")
        
        # Periodic save
        if (i + 1) % batch_save_interval == 0:
            save_results(results, output_dir, total)
    
    # Final save
    save_results(results, output_dir, total)
    
    if balance_exit:
        print("\n[SAVED] Checkpoint saved. Use --resume to continue.")
    
    return results


def save_results(results: List[Dict], output_dir: Path, total: int):
    """Save results and compute summary statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute stats
    stats = compute_fpr_stats(results)
    
    # Save raw results
    output_obj = {
        "dataset": "benign_1000",
        "total": total,
        "evaluated": len(results),
        "timestamp": datetime.now().isoformat(),
        "statistics": stats,
        "details": results,
    }
    
    with open(output_dir / "fpr_raw_results.json", 'w', encoding='utf-8') as f:
        json.dump(output_obj, f, indent=2, ensure_ascii=False)
    
    # Save summary
    with open(output_dir / "fpr_summary.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def compute_fpr_stats(results: List[Dict]) -> Dict[str, Any]:
    """Compute comprehensive FPR statistics from evaluation results."""
    
    # Filter out errors
    valid = [r for r in results if r['action'] != 'error']
    errors = [r for r in results if r['action'] == 'error']
    total = len(valid)
    
    if total == 0:
        return {"total": 0, "fpr": 0, "note": "no valid results"}
    
    # Overall FPR
    fp_count = sum(1 for r in valid if r['is_false_positive'])
    fpr = fp_count / total * 100
    
    # By domain
    by_domain = {}
    for domain in set(r['domain'] for r in valid):
        domain_results = [r for r in valid if r['domain'] == domain]
        domain_fp = sum(1 for r in domain_results if r['is_false_positive'])
        domain_total = len(domain_results)
        by_domain[domain] = {
            "total": domain_total,
            "false_positives": domain_fp,
            "fpr": domain_fp / domain_total * 100 if domain_total > 0 else 0,
        }
    
    # By modality
    by_modality = {}
    for mod in set(r['modality'] for r in valid):
        mod_results = [r for r in valid if r['modality'] == mod]
        mod_fp = sum(1 for r in mod_results if r['is_false_positive'])
        mod_total = len(mod_results)
        by_modality[mod] = {
            "total": mod_total,
            "false_positives": mod_fp,
            "fpr": mod_fp / mod_total * 100 if mod_total > 0 else 0,
        }
    
    # By defense source (which layer flagged it)
    by_source = {}
    for src in set(r['defense_source'] for r in valid):
        src_results = [r for r in valid if r['defense_source'] == src]
        src_fp = sum(1 for r in src_results if r['is_false_positive'])
        src_total = len(src_results)
        by_source[src] = {
            "total": src_total,
            "false_positives": src_fp,
            "fpr": src_fp / src_total * 100 if src_total > 0 else 0,
        }
    
    # By origin (MME, MM-Vet, custom)
    by_origin = {}
    for origin in set(r['source_origin'] for r in valid):
        origin_results = [r for r in valid if r['source_origin'] == origin]
        origin_fp = sum(1 for r in origin_results if r['is_false_positive'])
        origin_total = len(origin_results)
        by_origin[origin] = {
            "total": origin_total,
            "false_positives": origin_fp,
            "fpr": origin_fp / origin_total * 100 if origin_total > 0 else 0,
        }
    
    # Domain-Modality cross-tab
    domain_modality = {}
    for domain in set(r['domain'] for r in valid):
        domain_modality[domain] = {}
        for mod in set(r['modality'] for r in valid):
            cross = [r for r in valid if r['domain'] == domain and r['modality'] == mod]
            if cross:
                cross_fp = sum(1 for r in cross if r['is_false_positive'])
                domain_modality[domain][mod] = {
                    "total": len(cross),
                    "false_positives": cross_fp,
                    "fpr": cross_fp / len(cross) * 100,
                }
    
    # False positive details (for qualitative analysis)
    fp_details = [
        {
            "id": r['id'],
            "domain": r['domain'],
            "modality": r['modality'],
            "question": r['question'],
            "defense_source": r['defense_source'],
            "metadata": r.get('metadata_summary', {}),
        }
        for r in valid if r['is_false_positive']
    ]
    
    # Latency stats
    latencies = [r['elapsed_sec'] for r in valid if r.get('elapsed_sec', 0) > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    fp_latencies = [r['elapsed_sec'] for r in valid if r['is_false_positive'] and r.get('elapsed_sec', 0) > 0]
    avg_fp_latency = sum(fp_latencies) / len(fp_latencies) if fp_latencies else 0
    
    return {
        "total_evaluated": total,
        "total_errors": len(errors),
        "false_positives": fp_count,
        "fpr_percent": round(fpr, 2),
        "by_domain": by_domain,
        "by_modality": by_modality,
        "by_defense_source": by_source,
        "by_origin": by_origin,
        "domain_modality_cross": domain_modality,
        "false_positive_cases": fp_details,
        "latency": {
            "avg_sec": round(avg_latency, 2),
            "avg_fp_sec": round(avg_fp_latency, 2),
            "total_sec": round(sum(latencies), 2),
        },
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(stats: Dict):
    """Print a human-readable FPR report."""
    
    print(f"\n{'='*60}")
    print(f"=== FPR Evaluation Report ===")
    print(f"{'='*60}")
    print(f"Total evaluated: {stats['total_evaluated']}")
    print(f"Errors: {stats['total_errors']}")
    print(f"False Positives: {stats['false_positives']}")
    print(f"Overall FPR: {stats['fpr_percent']:.2f}%")
    
    print(f"\n--- By Domain ---")
    for domain, d in sorted(stats['by_domain'].items()):
        print(f"  {domain:12s}: FPR={d['fpr']:6.2f}% "
              f"(FP={d['false_positives']}/{d['total']})")
    
    print(f"\n--- By Modality ---")
    for mod, d in sorted(stats['by_modality'].items()):
        print(f"  {mod:20s}: FPR={d['fpr']:6.2f}% "
              f"(FP={d['false_positives']}/{d['total']})")
    
    print(f"\n--- By Defense Layer ---")
    for src, d in sorted(stats['by_defense_source'].items()):
        print(f"  {src:20s}: FPR={d['fpr']:6.2f}% "
              f"(FP={d['false_positives']}/{d['total']})")
    
    print(f"\n--- By Data Origin ---")
    for origin, d in sorted(stats['by_origin'].items()):
        print(f"  {origin:12s}: FPR={d['fpr']:6.2f}% "
              f"(FP={d['false_positives']}/{d['total']})")
    
    print(f"\n--- Latency ---")
    lat = stats['latency']
    print(f"  Average: {lat['avg_sec']:.2f}s")
    print(f"  Average FP: {lat['avg_fp_sec']:.2f}s")
    print(f"  Total: {lat['total_sec']:.0f}s ({lat['total_sec']/60:.1f} min)")
    
    if stats.get('false_positive_cases'):
        print(f"\n--- False Positive Cases ---")
        for fp in stats['false_positive_cases'][:20]:  # Show first 20
            print(f"  [{fp['id']}] {fp['domain']}/{fp['modality']}: "
                  f"layer={fp['defense_source']} | "
                  f"Q: {fp['question'][:60]}...")


def generate_latex_table(stats: Dict) -> str:
    """Generate LaTeX table for the paper."""
    
    lines = []
    
    # Table 1: Overall FPR
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{False Positive Rate on Benign Dataset "
                 r"($N=" + str(stats['total_evaluated']) + r"$).}")
    lines.append(r"\label{tab:fpr}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Domain} & \textbf{Samples} & "
                 r"\textbf{FPs} & \textbf{FPR (\%)} \\")
    lines.append(r"\midrule")
    
    for domain, d in sorted(stats['by_domain'].items()):
        lines.append(f"{domain.capitalize():12s} & {d['total']:4d} & "
                     f"{d['false_positives']:3d} & {d['fpr']:5.2f} \\\\")
    
    lines.append(r"\midrule")
    lines.append(r"\textbf{Overall} & "
                 f"{stats['total_evaluated']:4d} & "
                 f"{stats['false_positives']:3d} & "
                 f"\\textbf{{{stats['fpr_percent']:.2f}}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    # Table 2: FPR by defense layer
    lines.append("")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{False Positives by Defense Layer.}")
    lines.append(r"\label{tab:fpr_layer}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Layer} & \textbf{Samples} & "
                 r"\textbf{FPs} & \textbf{FPR (\%)} \\")
    lines.append(r"\midrule")
    
    for src, d in sorted(stats['by_defense_source'].items()):
        label = src.replace('_', ' ').title()
        lines.append(f"{label:20s} & {d['total']:4d} & "
                     f"{d['false_positives']:3d} & {d['fpr']:5.2f} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FPR on benign dataset using real DefenceOrchestrator"
    )
    parser.add_argument(
        "--dataset", type=str, default=DATASET_PATH,
        help="Path to benign dataset JSON"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max samples to evaluate (for pilot runs)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing results"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Do not resume (overwrite existing results)"
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Generate LaTeX tables after evaluation"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"=== Real FPR Evaluation ===")
    print(f"Dataset: {args.dataset}")
    samples = load_benign_dataset(args.dataset, args.limit)
    print(f"Samples loaded: {len(samples)}")
    print(f"Output: {output_dir}")
    
    # Run evaluation
    results = asyncio.run(run_evaluation(
        samples, output_dir,
        resume=not args.no_resume,
    ))
    
    # Compute and print stats
    stats = compute_fpr_stats(results)
    print_report(stats)
    
    # Generate LaTeX if requested
    if args.latex or True:  # Always generate
        latex = generate_latex_table(stats)
        latex_path = output_dir / "fpr_tables.tex"
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex)
        print(f"\nLaTeX tables saved to: {latex_path}")
    
    # Save final stats
    stats_path = output_dir / "fpr_summary.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {stats_path}")


if __name__ == "__main__":
    main()
