from __future__ import annotations
"""Evaluate MM-Vet predictions using the official Hugging Face Space API only.

This script does NOT run model inference; it only uploads an existing
predictions JSON (list[{"id": str, "answer": str}]) to the Space and returns
the official grading results.

Usage:
  python -m defenceAgent.eval.code.eval_mmvet_official_only \
    --preds d:\\AgentScope\\defenceAgent\\eval\\results\\mmvet_baseline_preds.json \
    --output d:\\AgentScope\\defenceAgent\\eval\\results\\mmvet_baseline_official_only.json

You can also pass OpenAI key/base/model explicitly or rely on env vars:
  - key:   --official-openai-key or env OPENAI_API_KEY / OAI_API_KEY
  - base:  --official-api-base   or env OPENAI_API_BASE / OPENAI_BASE_URL
  - model: --official-model (default: gpt-4.1)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

# 优先加载根目录与 defenceAgent 下的 .env，避免每次手动 export 环境变量
try:
  from dotenv import load_dotenv  # type: ignore
except Exception:
  load_dotenv = None  # 可选依赖

if load_dotenv is not None:
  # 先尝试加载仓库根目录下的 .env
  load_dotenv()
  # 再显式加载 defenceAgent/.env（不覆盖已有环境变量）
  da_env = Path(__file__).resolve().parents[2] / "defenceAgent" / ".env"
  if da_env.exists():
    load_dotenv(dotenv_path=str(da_env), override=False)

from defenceAgent.eval.code.mmvet_official_api import MMVetOfficialEvaluator


def main():
    ap = argparse.ArgumentParser(description="Official-only evaluator for MM-Vet predictions")
    ap.add_argument("--preds", required=True, help="Path to predictions JSON (list[{id, answer}])")
    ap.add_argument("--output", required=True, help="Where to save the official result JSON")
    ap.add_argument("--official-space", type=str, default="whyu/MM-Vet_Evaluator")
    ap.add_argument("--hf-token", type=str, default=None)
    ap.add_argument("--official-openai-key", type=str, default=None)
    ap.add_argument("--official-model", type=str, default="gpt-4.1", choices=["gpt-4.1","gpt-4-0613","gpt-4-turbo"])
    ap.add_argument("--official-api-base", type=str, default="")
    args = ap.parse_args()

    eva = MMVetOfficialEvaluator(space_id=args.official_space, hf_token=args.hf_token)
    result: Dict[str, Any] = eva.evaluate(
        predictions_json=args.preds,
        openai_key=(args.official_openai_key or None),
        model=args.official_model,
        api_base=args.official_api_base,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
