from __future__ import annotations
"""Evaluate HarmBench (text-only jailbreak prompts) using the DefenceOrchestrator.

Goal: demonstrate that the multimodal defence pipeline also works in single-modality (text-only) cases.

Metrics:
- Overall and per-category counts of allow/block/revise
- Refusal rate = 1 - allow/total
- Details include final action, rule source (if light_rules triggered), revised_text when applicable
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from agentscope.message import Msg, TextBlock

from defenceAgent.dataset_utils import load_harmbench_rows
from defenceAgent.orchestrator import DefenceOrchestrator
from defenceAgent.models import ModelRegistry
from defenceAgent.safety_classifier import classify_with_details


def build_user_msg_text(prompt: str, context: Optional[str]) -> Msg:
    text = prompt.strip()
    if context:
        # Put context first, then prompt
        text = f"Context:\n{context.strip()}\n\nPrompt:\n{text}"
    return Msg(name="user", role="user", content=[TextBlock(type="text", text=text)])


async def _gen_baseline_reply(user_msg: Msg) -> str:
    """Call text model directly to get an answer (no defence)."""
    reg = ModelRegistry()
    bundle = reg.build_text()
    sys_prompt = "You are a helpful assistant. Answer the user's request as best as you can."
    prompt = await bundle.formatter.format([
        Msg(name="system", role="system", content=sys_prompt),
        user_msg,
    ])
    resp = await bundle.model(prompt)
    content = resp.content
    # normalize content to plain text
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                parts.append(str(b.get("text")))
        return "\n".join(parts) if parts else str(content)
    return str(content)


def main():
    ap = argparse.ArgumentParser(description="Evaluate HarmBench with DefenceOrchestrator (text-only)")
    ap.add_argument("--dataset_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "statics" / "HarmBench"))
    ap.add_argument("--config", type=str, default="standard", choices=["standard", "contextual", "copyright"]) 
    ap.add_argument("--limit", type=int, default=0, help="<=0 means no limit")
    ap.add_argument("--mode", type=str, default="defence", choices=["defence", "baseline", "both"], help="Run defence pipeline, baseline, or both")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    rows = load_harmbench_rows(args.dataset_dir, config=args.config, limit=(None if args.limit is None or args.limit <= 0 else args.limit))

    def run_defence(rows_in: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        orch = DefenceOrchestrator()
        total = 0
        allow = block = revise = 0
        details: List[Dict[str, Any]] = []
        per_cat: Dict[str, Dict[str, int]] = {}

        for r in tqdm(rows_in, desc=f"HarmBench(defence):{args.config}"):
            rid = r.get("id")
            p = r.get("prompt") or ""
            ctx = r.get("context")
            cat = r.get("category") or "(uncategorized)"

            if not p.strip():
                continue

            user_msg = build_user_msg_text(p, ctx)
            try:
                final, img_res, txt_res = __import__("asyncio").run(orch.run(user_msg))
            except Exception as e:
                tqdm.write(f"[skip id={rid}] {type(e).__name__}: {e}")
                continue

            # Resolve and normalize final action
            meta = final.raw.metadata if final.raw else {}
            raw_action = None
            if isinstance(meta, dict):
                raw_action = meta.get("action")
            if raw_action is None:
                action = "allow" if final.safe else "block"
            else:
                ra = str(raw_action).lower()
                action = ra if ra in ("allow", "revise") else "block"

            total += 1
            if action == "allow":
                allow += 1
            elif action == "revise":
                revise += 1
            else:
                block += 1

            st = per_cat.setdefault(cat, {"total": 0, "allow": 0, "block": 0, "revise": 0})
            st["total"] += 1
            st[action] += 1

            rec: Dict[str, Any] = {
                "id": rid,
                "prompt": p,
                "context": ctx,
                "category": cat,
                "final_action": action,
                "final_safe": bool(final.safe),
            }
            if isinstance(meta, dict):
                rec["decision_meta"] = meta
                if meta.get("revised_text") and action == "revise":
                    rec["question_revised"] = meta.get("revised_text")
            details.append(rec)

        refusal_rate = 1.0 - (allow / total if total else 0.0)
        cat_breakdown: Dict[str, Dict[str, Any]] = {}
        for c, st in per_cat.items():
            t = st["total"]
            a = st["allow"]
            cat_breakdown[c] = {
                "total": t,
                "allow": a,
                "block": st["block"],
                "revise": st["revise"],
                "refusal_rate": (1.0 - a / t) if t else 0.0,
            }

        head = {
            "dataset": "HarmBench",
            "config": args.config,
            "mode": "defence",
            "total": total,
            "allow": allow,
            "block": block,
            "revise": revise,
            "refusal_rate": refusal_rate,
            "per_category": dict(sorted(cat_breakdown.items(), key=lambda kv: (-kv[1]["total"], kv[0]))),
        }
        return head, details

    def run_baseline(rows_in: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        total = 0
        allow = block = 0
        details: List[Dict[str, Any]] = []
        per_cat: Dict[str, Dict[str, int]] = {}

        for r in tqdm(rows_in, desc=f"HarmBench(baseline):{args.config}"):
            rid = r.get("id")
            p = r.get("prompt") or ""
            ctx = r.get("context")
            cat = r.get("category") or "(uncategorized)"
            if not p.strip():
                continue
            user_msg = build_user_msg_text(p, ctx)
            try:
                answer = __import__("asyncio").run(_gen_baseline_reply(user_msg))
            except Exception as e:
                tqdm.write(f"[skip id={rid}] {type(e).__name__}: {e}")
                continue

            # Classify answer safety; unsafe => model "allowed" harmful content
            try:
                is_unsafe, clf = classify_with_details(answer or "")
            except Exception:
                is_unsafe, clf = True, {"verdict": "unknown", "reasons": ["classifier_error"]}

            action = "allow" if is_unsafe else "block"
            total += 1
            if action == "allow":
                allow += 1
            else:
                block += 1

            st = per_cat.setdefault(cat, {"total": 0, "allow": 0, "block": 0, "revise": 0})
            st["total"] += 1
            st[action] += 1

            details.append({
                "id": rid,
                "prompt": p,
                "context": ctx,
                "category": cat,
                "final_action": action,
                "answer_preview": (answer[:2000] if isinstance(answer, str) else str(answer)) if answer else "",
                "classifier": clf,
            })

        refusal_rate = 1.0 - (allow / total if total else 0.0)
        cat_breakdown: Dict[str, Dict[str, Any]] = {}
        for c, st in per_cat.items():
            t = st["total"]
            a = st["allow"]
            cat_breakdown[c] = {
                "total": t,
                "allow": a,
                "block": st["block"],
                "revise": 0,
                "refusal_rate": (1.0 - a / t) if t else 0.0,
            }

        head = {
            "dataset": "HarmBench",
            "config": args.config,
            "mode": "baseline",
            "total": total,
            "allow": allow,
            "block": block,
            "revise": 0,
            "refusal_rate": refusal_rate,
            "per_category": dict(sorted(cat_breakdown.items(), key=lambda kv: (-kv[1]["total"], kv[0]))),
        }
        return head, details

    # Run according to mode
    if args.mode == "defence":
        head, details = run_defence(rows)
        report = {**head, "details": details}
    elif args.mode == "baseline":
        head, details = run_baseline(rows)
        report = {**head, "details": details}
    else:  # both
        b_head, b_details = run_baseline(rows)
        d_head, d_details = run_defence(rows)
        report = {
            "dataset": "HarmBench",
            "config": args.config,
            "mode": "both",
            "baseline": {**b_head, "details_count": len(b_details)},
            "defence": {**d_head, "details_count": len(d_details)},
            "summary": {
                "delta_refusal_rate": d_head.get("refusal_rate", 0.0) - b_head.get("refusal_rate", 0.0),
                "delta_allow": d_head.get("allow", 0) - b_head.get("allow", 0),
                "delta_block": d_head.get("block", 0) - b_head.get("block", 0),
                "delta_revise": d_head.get("revise", 0) - b_head.get("revise", 0),
            },
        }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
