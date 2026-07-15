from __future__ import annotations
import argparse
import base64
from pathlib import Path
from typing import Optional

from agentscope.message import Msg, ImageBlock, TextBlock, Base64Source, URLSource

from .orchestrator import DefenceOrchestrator
from .models import ModelRegistry
from .executor import MLLMExecutor


def load_image_source(path_or_url: str):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return URLSource(type="url", url=path_or_url)
    p = Path(path_or_url)
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    # naive type guess
    mt = "image/png" if p.suffix.lower() in {".png"} else "image/jpeg"
    return Base64Source(type="base64", media_type=mt, data=b64)


def build_user_msg(image: Optional[str], text: Optional[str]) -> Msg:
    blocks = []
    if text:
        blocks.append(TextBlock(type="text", text=text))
    if image:
        blocks.append(ImageBlock(type="image", source=load_image_source(image)))
    if not blocks:
        raise ValueError("必须至少提供 --text 或 --image 之一")
    return Msg(name="user", role="user", content=blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description="DefenceAgent demo")
    parser.add_argument("--image", type=str, default=None, help="本地图片路径或URL")
    parser.add_argument("--text", type=str, default=None, help="文本输入")
    args = parser.parse_args()

    # warm up registry (also validates env presence at import time)
    _ = ModelRegistry()

    user_msg = build_user_msg(args.image, args.text)
    orchestrator = DefenceOrchestrator()

    import asyncio

    final, img_res, txt_res = asyncio.run(orchestrator.run(user_msg))

    def pretty(res, title):
        meta = res.raw.metadata
        print(f"\n== {title} ==")
        print("safe:", res.safe)
        print("metadata:", meta)

    if img_res:
        pretty(img_res, "ImageSafety")
    if txt_res:
        pretty(txt_res, "TextSafety")

    print("\n== Final (Judge) ==")
    print("source:", final.source)
    print("safe:", final.safe)
    print("metadata:", final.raw.metadata)

    # Call final MLLM based on judge result
    executor = MLLMExecutor()
    final_msg = asyncio.run(executor(user_msg, final.raw.metadata if final.raw else None))
    print("\n== MLLM Output ==")
    print(final_msg.content)


if __name__ == "__main__":
    main()
