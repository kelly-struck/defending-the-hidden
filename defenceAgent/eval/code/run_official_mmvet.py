from __future__ import annotations
"""
仅调用官方 MM-Vet 评估器（Hugging Face Space）为已有预测文件打分。

用法（示例）：
  python -m defenceAgent.eval.code.run_official_mmvet \
    --preds d:\\AgentScope\\defenceAgent\\eval\\results\\mmvet_baseline_preds.json \
    --output d:\\AgentScope\\defenceAgent\\eval\\results\\mmvet_baseline_official.json \
    --official-model gpt-4.1 --official-api-base %OPENAI_API_BASE%
"""

import argparse
import json
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from defenceAgent.eval.code.mmvet_official_api import MMVetOfficialEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run official MM-Vet evaluator on an existing predictions file")
    parser.add_argument("--preds", required=True, type=str, help="预测文件路径（list[{id, answer}]）")
    parser.add_argument("--output", type=str, default=None, help="保存官方评分 JSON 的路径")
    parser.add_argument("--official-space", type=str, default="whyu/MM-Vet_Evaluator", help="Space 标识（owner/name）")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face 访问令牌（可选）")
    parser.add_argument("--official-openai-key", type=str, default=None, help="Space 使用的 OpenAI API Key（可选，默认读取 OPENAI_API_KEY）")
    parser.add_argument("--official-model", type=str, default="gpt-4.1", choices=["gpt-4-0613","gpt-4-turbo","gpt-4.1"], help="评分所用模型")
    parser.add_argument("--official-api-base", type=str, default="", help="OpenAI 兼容 API Base（可选）")
    parser.add_argument("--retries", type=int, default=3, help="失败重试次数（默认3）")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="重试退避系数，指数退避（默认2.0秒）")

    args = parser.parse_args()

    eva = MMVetOfficialEvaluator(space_id=args.official_space, hf_token=args.hf_token, max_retries=args.retries, retry_backoff_sec=args.retry_backoff)
    try:
        scores = eva.evaluate(
            args.preds,
            openai_key=(args.official_openai_key or None),
            model=args.official_model,
            api_base=args.official_api_base,
        )
    except Exception as e:
        print(f"[official_evaluator] 调用失败: {type(e).__name__}: {e}")
        return

    text = json.dumps(scores, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
