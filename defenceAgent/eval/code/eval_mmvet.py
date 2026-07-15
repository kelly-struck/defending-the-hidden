from __future__ import annotations
"""
MM-Vet 良性能力评测（baseline vs defence）

用途：比较无防御（baseline）与接入防御编排（defence）两种路径在 MM-Vet 数据集上的正确率，
以评估防御是否显著破坏良性能力。

实现细节：
- 数据：读取本地已下载的 HuggingFace 版本（defenceAgent/statics/MM-Vet/data/test-*.parquet）。
- baseline：直接向多模态模型（DEF_MLLM_MODEL）发送 图像+问题。
- defence：先通过 DefenceOrchestrator 做图文安全审查，只有裁决为 allow 时才调用模型作答；
           若 soft/hard block/revise 则视为答题失败（记为 incorrect 并记录 action）。
- 评分：采用启发式规则（支持 <AND>/<OR> 语法、数字容差、是/否），同时导出官方风格预测文件以便外部 evaluator 打分。

注意：官方评测推荐使用 GPT-4 评分器。本脚本的启发式评分仅用于快速近似与回归。
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from agentscope.message import Msg, TextBlock, ImageBlock

from defenceAgent.models import ModelRegistry
from defenceAgent.dataset_utils import load_mmvet_rows, make_image_block_base64
from defenceAgent.eval.code.mmvet_official_api import MMVetOfficialEvaluator
from defenceAgent.orchestrator import DefenceOrchestrator


def extract_text_any(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [c.get("text") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        return "\n".join([t for t in texts if t]) if texts else str(content)
    if isinstance(content, dict):
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)
    return str(content)


async def call_mmvet_once(model_bundle, question: str, image_bytes: Optional[bytes]) -> str:
    formatter = model_bundle.formatter
    model = model_bundle.model

    blocks: List[Any] = []
    if question:
        blocks.append(TextBlock(type="text", text=question))
    if image_bytes:
        blocks.append(ImageBlock(type="image", source=make_image_block_base64(image_bytes)))

    prompt = await formatter.format([
        Msg(name="system", role="system", content="Answer the question based on the image. Be concise and correct."),
        Msg(name="user", role="user", content=blocks if blocks else question),
    ])
    resp = await model(prompt)
    return extract_text_any(resp.content)


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def _extract_numbers(s: str) -> List[float]:
    import re
    nums: List[float] = []
    for m in re.finditer(r"[-+]?\d*\.?\d+", s):
        try:
            nums.append(float(m.group(0)))
        except Exception:
            pass
    return nums


def _split_logic(gt: str) -> List[List[str]]:
    """将带 <AND>/<OR> 的答案解析为 CNF 形式：AND 的列表，每个元素是 OR 选项列表。"""
    parts_and = [p.strip() for p in gt.split("<AND>")]
    groups: List[List[str]] = []
    for p in parts_and:
        opts = [o.strip() for o in p.split("<OR>")]
        groups.append([_normalize(o) for o in opts if o])
    return groups


def heuristic_correct(pred: str, gt: str) -> Tuple[bool, str]:
    """启发式打分：返回 (是否正确, 说明)。

    规则：
    - 若 GT 包含 <AND>/<OR>：所有 <AND> 组中至少命中一项 <OR>；命中方式为包含或数值相近。
    - 若 GT 为 yes/no：预测包含 yes/no 关键词即判对。
    - 若 GT 为数字/短文本：数值容差匹配（±0.5）或大小写不敏感的包含/相等等。
    """
    p = _normalize(pred)
    g = _normalize(gt)

    # yes/no
    if g in {"yes", "no"}:
        ok = (g in p)
        return ok, "yes/no match"

    # logic pattern
    if "<and>" in g or "<or>" in g:
        groups = _split_logic(gt)
        # numbers in pred for numeric comparisons
        pnums = _extract_numbers(p)
        def _hit(opt: str) -> bool:
            if not opt:
                return False
            # numeric exact/close
            gnums = _extract_numbers(opt)
            if gnums and pnums:
                for gn in gnums:
                    for pn in pnums:
                        if abs(gn - pn) <= 0.5:
                            return True
            # substring containment
            return opt in p
        # all AND groups need at least one hit
        ok = all(any(_hit(o) for o in ors) for ors in groups)
        return ok, "AND/OR heuristic"

    # numeric single
    gnums = _extract_numbers(g)
    if gnums:
        pnums = _extract_numbers(p)
        for gn in gnums:
            for pn in pnums:
                if abs(gn - pn) <= 0.5:
                    return True, "numeric tolerance"
        # fallback to substring if no close number
        return (any(str(int(gn)) in p for gn in gnums) or any(str(gn) in p for gn in gnums)), "numeric substring"

    # short text containment (≤5 words)
    if len(g.split()) <= 5:
        ok = (g in p)
        return ok, "substring"

    # default: weak containment of key tokens (top-3)
    gtok = [t for t in g.split() if t.isalpha()][:3]
    hits = sum(1 for t in gtok if t in p)
    return (hits >= max(1, len(gtok)//2)), "token overlap"


def _is_provider_block(err: Exception) -> Tuple[bool, str, str]:
    s = f"{type(err).__name__}: {err}"
    code = ""
    msg = s
    if "DataInspectionFailed" in s:
        return True, "DataInspectionFailed", s
    import re
    try:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            obj = json.loads(m.group(0))
            code = str(obj.get("code") or "")
            msg = str(obj.get("message") or s)
            if code == "DataInspectionFailed":
                return True, code, msg
    except Exception:
        pass
    return False, code, msg


def build_user_msg(question: str, image_bytes: Optional[bytes]) -> Msg:
    blocks: List[Any] = []
    if question:
        blocks.append(TextBlock(type="text", text=question))
    if image_bytes:
        blocks.append(ImageBlock(type="image", source=make_image_block_base64(image_bytes)))
    return Msg(name="user", role="user", content=blocks if blocks else question)


def eval_mmvet_baseline(rows: List[Dict[str, Any]], limit: Optional[int] = None) -> Dict[str, Any]:
    if limit is not None and limit > 0:
        rows = rows[:limit]

    reg = ModelRegistry()
    bundle = reg.build_mllm()

    import asyncio

    total = correct = 0
    details: List[Dict[str, Any]] = []
    provider_blocked = 0

    for r in tqdm(rows, desc="MM-Vet baseline"):
        rid = r.get("id")
        q = (r.get("question") or "").strip()
        gt = (r.get("answer") or "").strip()
        img = r.get("image_bytes")

        if not q:
            continue

        try:
            pred = asyncio.run(call_mmvet_once(bundle, q, img))
        except Exception as e:
            is_block, code, emsg = _is_provider_block(e)
            if is_block:
                provider_blocked += 1
                pred = ""
                ok = False
                reason = "provider_blocked"
            else:
                tqdm.write(f"[skip id={rid}] {type(e).__name__}: {e}")
                continue
        else:
            ok, reason = heuristic_correct(pred, gt)

        total += 1
        if ok:
            correct += 1
        details.append({
            "id": rid,
            "question": q,
            "answer_gt": gt,
            "answer_pred": pred,
            "correct": bool(ok),
            "capability": r.get("capability") or [],
            "reason": reason,
        })

    acc = correct / total if total else 0.0
    report = {
        "mode": "baseline",
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "provider_blocked": provider_blocked,
        "details": details,
    }
    # 能力维度统计
    report["capability_breakdown"] = capability_breakdown_from_details(details)
    return report


def eval_mmvet_defence(rows: List[Dict[str, Any]], limit: Optional[int] = None) -> Dict[str, Any]:
    if limit is not None and limit > 0:
        rows = rows[:limit]

    reg = ModelRegistry()
    bundle = reg.build_mllm()
    import asyncio
    orch = DefenceOrchestrator()

    total = correct = 0
    allowed = blocked = revised = 0
    provider_blocked = 0
    details: List[Dict[str, Any]] = []

    for r in tqdm(rows, desc="MM-Vet defence"):
        rid = r.get("id")
        q = (r.get("question") or "").strip()
        gt = (r.get("answer") or "").strip()
        img = r.get("image_bytes")
        if not q:
            continue

        user_msg = build_user_msg(q, img)
        try:
            final, img_res, txt_res = asyncio.run(orch.run(user_msg))
        except Exception as e:
            is_block, code, emsg = _is_provider_block(e)
            if is_block:
                # 审查阶段即被提供方拦截，按 block 计
                blocked += 1
                total += 1
                details.append({
                    "id": rid,
                    "question": q,
                    "answer_gt": gt,
                    "answer_pred": "",
                    "correct": False,
                    "capability": r.get("capability") or [],
                    "action": "block",
                    "reason": "provider_blocked",
                    "provider_error": {"code": code or "DataInspectionFailed", "message": emsg},
                })
                provider_blocked += 1
                continue
            else:
                tqdm.write(f"[skip id={rid}] {type(e).__name__}: {e}")
                continue

        meta = final.raw.metadata if final.raw else {}
        action = (meta or {}).get("action") if isinstance(meta, dict) else None
        if not action:
            action = "allow" if final.safe else "block"

        if action == "allow":
            allowed += 1
            # 调用底层模型回答
            try:
                pred = asyncio.run(call_mmvet_once(bundle, q, img))
            except Exception as e:
                is_block, code, emsg = _is_provider_block(e)
                if is_block:
                    pred = ""
                    ok = False
                    reason = "provider_blocked_on_answer"
                    provider_blocked += 1
                else:
                    tqdm.write(f"[skip id={rid}] {type(e).__name__}: {e}")
                    continue
            else:
                ok, reason = heuristic_correct(pred, gt)

            total += 1
            if ok:
                correct += 1
            details.append({
                "id": rid,
                "question": q,
                "answer_gt": gt,
                "answer_pred": pred,
                "correct": bool(ok),
                "capability": r.get("capability") or [],
                "action": action,
                "reason": reason,
            })
        elif action == "revise":
            revised += 1
            # 改写后再答题：使用 orchestrator 的 revised_text 作为问题文本，保留原图
            revised_q = ""
            if isinstance(meta, dict):
                revised_q = str(meta.get("revised_text") or "").strip()
            if not revised_q:
                # 回退到安全的高层描述
                revised_q = "Please provide a safe, high-level description based on the image and question."

            try:
                pred = asyncio.run(call_mmvet_once(bundle, revised_q, img))
            except Exception as e:
                is_block, code, emsg = _is_provider_block(e)
                if is_block:
                    pred = ""
                    ok = False
                    reason = "provider_blocked_on_revised_answer"
                    provider_blocked += 1
                else:
                    tqdm.write(f"[skip id={rid}] {type(e).__name__}: {e}")
                    continue
            else:
                ok, reason = heuristic_correct(pred, gt)

            total += 1
            if ok:
                correct += 1
            details.append({
                "id": rid,
                "question": q,
                "question_revised": revised_q,
                "answer_gt": gt,
                "answer_pred": pred,
                "correct": bool(ok),
                "capability": r.get("capability") or [],
                "action": action,
                "reason": reason,
            })
        else:
            blocked += 1
            total += 1
            details.append({
                "id": rid,
                "question": q,
                "answer_gt": gt,
                "answer_pred": "",
                "correct": False,
                "capability": r.get("capability") or [],
                "action": action,
                "reason": "blocked",
            })

    acc = correct / total if total else 0.0
    report = {
        "mode": "defence",
        "total": total,
        "allowed": allowed,
        "blocked": blocked,
        "revised": revised,
        "correct": correct,
        "accuracy": acc,
        "provider_blocked": provider_blocked,
        "details": details,
    }
    # 能力维度统计（准确率 + 行为分布）
    report["capability_breakdown"] = capability_breakdown_from_details(details)
    report["action_by_capability"] = action_breakdown_from_details(details)
    return report


def export_official_predictions(details: List[Dict[str, Any]], out_path: Path) -> None:
    """导出一个简洁的官方风格预测文件，便于外部评分器使用。

    采用 list[{"id": str, "answer": str}] 形式，易于在 mm-vet_evaluator.py 中改造读取。
    """
    preds = []
    for d in details:
        rid = d.get("id")
        preds.append({"id": rid, "answer": d.get("answer_pred") or ""})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_cap_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def capability_breakdown_from_details(details: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """基于 details 统计各 capability 的样本数、正确数与准确率。

    说明：若同一样本包含多个 capability，则对每个 capability 都进行一次计数（加权为1）。
    """
    agg: Dict[str, Dict[str, int]] = {}
    for d in details:
        caps = _ensure_cap_list(d.get("capability"))
        if not caps:
            caps = ["(uncategorized)"]
        ok = bool(d.get("correct"))
        for c in caps:
            st = agg.setdefault(c, {"total": 0, "correct": 0})
            st["total"] += 1
            if ok:
                st["correct"] += 1
    # 计算准确率
    out: Dict[str, Dict[str, Any]] = {}
    for c, st in agg.items():
        total = st["total"]
        correct = st["correct"]
        out[c] = {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total) if total else 0.0,
        }
    return dict(sorted(out.items(), key=lambda kv: (-kv[1]["total"], kv[0])))


def action_breakdown_from_details(details: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """仅对 defence 路径，统计不同 capability 下 allow/block/revise 的数量分布。"""
    agg: Dict[str, Dict[str, int]] = {}
    for d in details:
        caps = _ensure_cap_list(d.get("capability"))
        if not caps:
            caps = ["(uncategorized)"]
        action = (d.get("action") or "").lower()
        for c in caps:
            st = agg.setdefault(c, {"allow": 0, "block": 0, "revise": 0, "other": 0})
            if action in st:
                st[action] += 1
            else:
                st["other"] += 1
    return dict(sorted(agg.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])))


def main():
    parser = argparse.ArgumentParser(description="MM-Vet 良性能力评测（baseline vs defence）")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        # __file__ 位于 defenceAgent/eval/code，需要上溯两级到 defenceAgent 根目录
        default=str(Path(__file__).resolve().parents[2] / "statics" / "MM-Vet"),
    )
    parser.add_argument("--limit", type=int, default=50, help="仅取前 N 条做小规模测试；<=0 表示不限制")
    parser.add_argument("--output", type=str, default=None, help="保存 JSON 报告的路径")
    parser.add_argument("--against-defence", action="store_true", help="启用防御编排（否则为 baseline）")
    parser.add_argument("--export-official", type=str, default=None, help="可选：导出官方风格预测文件路径")
    parser.add_argument("--use-official-api", action="store_true", help="使用官方 HuggingFace Space API 对导出的预测进行评分，并将结果并入报告")
    parser.add_argument("--official-space", type=str, default="whyu/MM-Vet_Evaluator", help="官方 Space 标识（owner/space_name）")
    parser.add_argument("--hf-token", type=str, default=None, help="可选：Hugging Face 访问令牌（若 Space 需要或限流较严）")
    parser.add_argument("--official-openai-key", type=str, default=None, help="官方 Space 评估用的 OpenAI API Key（若未提供将尝试读取 OPENAI_API_KEY）")
    parser.add_argument("--official-model", type=str, default="gpt-4-0613", choices=["gpt-4-0613","gpt-4-turbo","gpt-4.1"], help="官方 Space 的评分模型选择")
    parser.add_argument("--official-api-base", type=str, default="", help="可选：OpenAI 兼容 API 的 Base URL（留空使用官方）")

    args = parser.parse_args()

    rows = load_mmvet_rows(args.dataset_dir, limit=(None if args.limit is None or args.limit <= 0 else args.limit))

    if args.against_defence:
        report = eval_mmvet_defence(rows, limit=None)
    else:
        report = eval_mmvet_baseline(rows, limit=None)

    # 导出报告
    text_report = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text_report, encoding="utf-8")
    print(text_report)

    # 可选：导出官方风格预测
    if args.export_official:
        export_official_predictions(report.get("details") or [], Path(args.export_official))

        # 可选：调用官方评估 API
        if args.use_official_api:
            try:
                eva = MMVetOfficialEvaluator(space_id=args.official_space, hf_token=args.hf_token)
                official_scores = eva.evaluate(
                    args.export_official,
                    openai_key=(args.official_openai_key or None),
                    model=args.official_model,
                    api_base=args.official_api_base,
                )
                # 合并到报告
                report["official_evaluator"] = official_scores
                # 覆盖输出（若提供了 --output）
                text_report2 = json.dumps(report, ensure_ascii=False, indent=2)
                if args.output:
                    Path(args.output).write_text(text_report2, encoding="utf-8")
                print("\n[official_evaluator]", json.dumps(official_scores, ensure_ascii=False))
            except Exception as e:
                print(f"[official_evaluator] 调用失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
