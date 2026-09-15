from __future__ import annotations
"""MM-Vet 官方评估 API 客户端（Hugging Face Spaces: whyu/MM-Vet_Evaluator）

依赖：gradio_client

典型使用：
  from mmvet_official_api import MMVetOfficialEvaluator
  eva = MMVetOfficialEvaluator(space_id="whyu/MM-Vet_Evaluator", hf_token=os.getenv("HF_TOKEN"))
  result = eva.evaluate(preds_json_path)

注意：不同时间点 Space 的 api_name 可能不同。我们将自动探测端点，优先尝试常见名称
      ['/evaluate', '/predict', '/run_evaluate']，否则回退到第一个可用端点。
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional

from gradio_client import Client, handle_file


class MMVetOfficialEvaluator:
    def __init__(self, space_id: str = "whyu/MM-Vet_Evaluator", hf_token: Optional[str] = None, *, max_retries: int = 3, retry_backoff_sec: float = 2.0) -> None:
        self.space_id = space_id
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        # lazy init of client
        self._client: Optional[Client] = None
        # retry config
        try:
            self.max_retries = int(os.getenv("MMVET_OFFICIAL_MAX_RETRIES", str(max_retries)))
        except Exception:
            self.max_retries = max_retries
        try:
            self.retry_backoff_sec = float(os.getenv("MMVET_OFFICIAL_RETRY_BACKOFF", str(retry_backoff_sec)))
        except Exception:
            self.retry_backoff_sec = retry_backoff_sec

    def _ensure_client(self) -> Client:
        if self._client is None:
            if self.hf_token:
                self._client = Client(self.space_id, hf_token=self.hf_token)
            else:
                self._client = Client(self.space_id)
        return self._client

    def _pick_api_name(self) -> str:
        client = self._ensure_client()
        try:
            apis = client.view_api(all_endpoints=True)
        except Exception:
            # fallback to a common default
            return "/run_grade"
        # Some client versions may return None here; fallback to /run_grade
        if not apis:
            return "/run_grade"

        # Find common names first
        # Strongly prefer the observed endpoint in Space usage info
        preferred = ["/run_grade", "/evaluate", "/predict", "/run_evaluate", "/run/predict", "/_toggle_fields"]
        names = [getattr(a, "api_name", None) or getattr(a, "path", None) for a in apis]
        names = [n for n in names if isinstance(n, str)]
        for want in preferred:
            if want in names:
                return want
        # fallback to first available
        return names[0] if names else "/run_grade"

    def evaluate(
        self,
        predictions_json: str | Path,
        *,
        openai_key: Optional[str] = None,
        model: str = "gpt-4.1",
        api_base: str = "",
    ) -> Dict[str, Any]:
        """上传本地预测文件到 Space，返回官方评分结果。

        参数 predictions_json: 形如 list[{"id": str, "answer": str}] 的 JSON 文件路径。
        返回：若 Space 返回文件，下载并尽力解析为 JSON；否则返回原始对象或文件路径。
        """
        predictions_json = str(predictions_json)
        client = self._ensure_client()
        api_name = self._pick_api_name()

        # 获取 OpenAI Key / Base：优先参数，其次环境变量
        # Key: OPENAI_API_KEY / OAI_API_KEY
        key = openai_key or os.getenv("OPENAI_API_KEY") or os.getenv("OAI_API_KEY") or ""
        # Base: 显式传入 > OPENAI_API_BASE > OPENAI_BASE_URL > 空
        base = api_base if api_base else (os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "")

        # gradio_client 会自动处理文件上传。不同 Space 可能需要额外参数；
        # 若失败，请查看 Space 的 UI 输入项，并在此处按需扩展参数。
        last_err: Optional[Exception] = None
        for attempt in range(max(1, self.max_retries)):
            try:
                if api_name == "/run_grade":
                    # According to Space usage: predict(file_obj, key, model, api_base, api_name="/run_grade") -> File
                    out = client.predict(
                        file_obj=handle_file(predictions_json),
                        key=key,
                        model=model,
                        api_base=base,
                        api_name=api_name,
                    )
                else:
                    # Fallback: try best-effort variants
                    try:
                        out = client.predict(predictions_json, api_name=api_name)
                    except TypeError:
                        out = client.predict(result_file=predictions_json, api_name=api_name)
                break  # success
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    # 指数退避后重试
                    time.sleep(self.retry_backoff_sec * (2 ** attempt))
                    continue
                # 附带一次 API 概览，帮助诊断（仅最后一次失败时）
                try:
                    info = client.view_api(all_endpoints=True)
                except Exception:
                    info = None
                raise RuntimeError(f"Call failed for api_name={api_name}: {e}; endpoints={info}")

        # 结果通常是可 JSON 化的结构或字符串
        if isinstance(out, (str, bytes)):
            s = out.decode("utf-8", errors="ignore") if isinstance(out, (bytes, bytearray)) else str(out)
            # if looks like a local file path, try to load
            p = Path(s)
            if p.exists() and p.is_file():
                # try json
                try:
                    txt = p.read_text(encoding="utf-8")
                    return json.loads(txt)
                except Exception:
                    # not json; return file info
                    return {"download": str(p), "note": "non-json file returned (e.g., zip)"}
            # else try parse json from string
            try:
                return json.loads(s)
            except Exception:
                return {"raw": s}
        if isinstance(out, (list, tuple)) and len(out) == 1:
            v = out[0]
            if isinstance(v, (str, bytes)):
                s = v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else str(v)
                p = Path(s)
                if p.exists() and p.is_file():
                    try:
                        txt = p.read_text(encoding="utf-8")
                        return json.loads(txt)
                    except Exception:
                        return {"download": str(p), "note": "non-json file returned (e.g., zip)"}
                try:
                    return json.loads(s)
                except Exception:
                    return {"raw": s}
        # fallback
        try:
            json.dumps(out)
            return out  # type: ignore
        except Exception:
            return {"raw": str(out)}
