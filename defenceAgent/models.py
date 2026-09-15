from __future__ import annotations
import os
from typing import Any, List
from dataclasses import dataclass

from pydantic import BaseModel
from dotenv import load_dotenv
from agentscope.model import DashScopeChatModel, OpenAIChatModel
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg, TextBlock, ImageBlock, URLSource, Base64Source

# load environment variables from .env if present
load_dotenv()


@dataclass
class ModelBundle:
    """Aggregate a chat model and its formatter for convenience."""
    model: Any
    formatter: Any


class ModelRegistry(BaseModel):
    """Simple registry to construct models from env vars.

    You can swap out DashScopeChatModel to your own model adapter if needed.
    """

    mllm_model_name: str = os.getenv("DEF_MLLM_MODEL", "qwen-vl-plus")
    text_model_name: str = os.getenv("DEF_TEXT_MODEL", "qwen2.5-7b-instruct")
    image_model_name: str = os.getenv("DEF_IMAGE_MODEL", "qwen2-vl-7b-instruct")
    target_model_name: str = os.getenv("DEF_TARGET_MODEL", "doubao-1.5-vision-pro-250328")
    api_key: str | None = os.getenv("DASHSCOPE_API_KEY")
    # Phase 1 (E1): temperature 支持，供 C1 三种子（3 次独立运行 t=0.7）与消融温度扫描。
    # 注：DashScope 的 seed 参数对 Qwen 系列不保证可复现，故 C1 以 3 次独立完整运行实现。
    temperature: float = 0.7
    seed: int | None = None

    def build_mllm(self) -> ModelBundle:
        model = DashScopeChatModel(
            model_name=self.mllm_model_name, api_key=self.api_key, stream=False,
            generate_kwargs=self._gen_kwargs(),
        )
        return ModelBundle(model=model, formatter=DashScopeChatFormatter())

    def build_target_mllm(self) -> ModelBundle:
        """Build the target model (defended model)."""
        print(f"ModelRegistry.build_target_mllm called with target_model_name='{self.target_model_name}'")
        if "doubao" in self.target_model_name.lower():
            print("Using DoubaoModel")
            from .doubao_wrapper import DoubaoModel, IdentityFormatter
            # Doubao uses ARK_API_KEY, usually handled inside the wrapper or env
            model = DoubaoModel(config_name=self.target_model_name)
            # Use IdentityFormatter to pass Msg objects directly to DoubaoModel
            # DashScopeChatFormatter converts to DashScope dicts which DoubaoModel doesn't parse correctly
            return ModelBundle(model=model, formatter=IdentityFormatter())
        else:
            print(f"Using DashScopeChatModel for {self.target_model_name}")
            # Fallback to default MLLM if not doubao
            model = DashScopeChatModel(
                model_name=self.target_model_name, api_key=self.api_key, stream=False,
                generate_kwargs=self._gen_kwargs(),
            )
            return ModelBundle(model=model, formatter=DashScopeChatFormatter())

    def build_text(self) -> ModelBundle:
        model = DashScopeChatModel(
            model_name=self.text_model_name, api_key=self.api_key, stream=False,
            generate_kwargs=self._gen_kwargs(),
        )
        return ModelBundle(model=model, formatter=DashScopeChatFormatter())

    def build_image(self) -> ModelBundle:
        model = DashScopeChatModel(
            model_name=self.image_model_name, api_key=self.api_key, stream=False,
            generate_kwargs=self._gen_kwargs(),
        )
        return ModelBundle(model=model, formatter=DashScopeChatFormatter())

    def _gen_kwargs(self) -> dict:
        kwargs: dict = {"temperature": self.temperature}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs


# Utilities to help construct multimodal messages

def text_msg(system_prompt: str, user_text: str, formatter: Any) -> List[Msg]:
    return [
        Msg(name="system", role="system", content=system_prompt),
        Msg(name="user", role="user", content=user_text),
    ]


def image_text_msg(system_prompt: str, user_text: str | None, image_source: URLSource | Base64Source, formatter: Any) -> List[Msg]:
    blocks = []
    if user_text:
        blocks.append(TextBlock(type="text", text=user_text))
    blocks.append(ImageBlock(type="image", source=image_source))
    return [
        Msg(name="system", role="system", content=system_prompt),
        Msg(name="user", role="user", content=blocks),
    ]
