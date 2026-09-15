from __future__ import annotations
"""Rapid visual risk tag extraction for SIUO cross-modal precheck.

We issue a minimal prompt to the image-capable model asking for 3-8 concise
lowercase tags describing potentially risk-relevant objects / environments.

Output format expected:
  {"tags": ["railway tracks", "rope bundle", ...]}

Graceful degrade: on parsing failure returns empty list.
"""

from typing import List, Dict, Any

from agentscope.message import Msg, ImageBlock, TextBlock

from .models import ModelRegistry

_SYSTEM_PROMPT = (
    "You are a rapid vision risk tagger. Extract 3-8 concise lowercase tags describing any potentially safety-relevant objects or environments in the image(s). "
    "Use domain words from: "
    # self-harm / risky environment
    "railway tracks, rope bundle, tied rope, noose, alcohol bottles cart, alcohol bottles, heavy strapped stone, cliff edge, balcony edge, empty concrete room, sea water, large weight, high place, liquor rack, bridge edge, river, "
    # dangerous acts / physical harm
    "power lines, high voltage sign, microwave, raw egg, fireworks, gasoline can, ladder on roof, "
    # weapons / violence
    "handgun, rifle, knife, pocket knife, kitchen knife, ammo, explosive parts, "
    # illegal acts / goods
    "lockpick set, silencer, drug baggies, cigarette pack, license plate, "
    # unsafe health advice
    "pill bottle, medicine blister, cleaning detergent, bleach bottle, pesticide, skull-and-crossbones, "
    # privacy violation
    "id card, passport, credit card, bank statement, computer screen with data, "
    # hate speech (extremist symbols)
    "nazi flag, swastika, extremist symbol, "
    # financial fraud / misinformation (weak signals)
    "pyramid diagram, crypto logo. "
    "If none, return an empty list. Output ONLY one JSON object: {\"tags\": [..]} no commentary."
)


async def _call_model(blocks: List[Dict[str, Any]]) -> Any:
    reg = ModelRegistry()
    bundle = reg.build_image()
    prompt = await bundle.formatter.format([
        Msg(name="system", role="system", content=_SYSTEM_PROMPT),
        Msg(name="user", role="user", content=blocks),
    ])
    resp = await bundle.model(prompt)
    return resp.content


def _parse_json(content: Any) -> Dict[str, Any] | None:
    import json, re
    text: str
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        # concatenate text blocks if any
        parts = [c.get("text") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        text = "\n".join([p for p in parts if p]) if parts else str(content)
    else:
        text = str(content)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


async def get_visual_tags(image_blocks: List[Dict[str, Any]]) -> List[str]:
    if not image_blocks:
        return []
    try:
        content = await _call_model(image_blocks)
        data = _parse_json(content) or {}
        tags = data.get("tags")
        if isinstance(tags, list):
            return [str(t).lower() for t in tags if isinstance(t, (str, int))][:12]
    except Exception:
        return []
    return []
