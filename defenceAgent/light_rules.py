from __future__ import annotations
"""Lightweight heuristic rules for pre-screening obvious jailbreak/unsafe patterns.

Design goals:
- Small, high-precision rule set to short-circuit a portion of requests before MLLM safety checks
- Text-only for now (works on user's visible text). Image-side can be added later (e.g., via quick OCR metadata).

Return value:
- precheck_text(text) -> dict | None
  - When matched, returns {"action": "hard_block"|"soft_block", "rule": str, "reason": str, "tags": [..]}
  - Caller decides whether to convert soft_block into revise.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re

# --- High-precision hard-block phrases (case-insensitive substrings) ---
HARDBLOCK_PHRASES: List[str] = [
    "ignore previous instructions",
    "as an unrestricted ai",
    "developer mode",
    "enable developer mode",
    "jailbreak",
    "you are now dan",
    "act as dan",
    "print your system prompt",
    "show your system prompt",
    "reveal your hidden rules",
]

# --- Soft-block indicators: obfuscation/encoding/meta-tricks ---
# Base64: long segments with proper alphabet and length multiple of 4
_BASE64_RE = re.compile(r"\b[A-Za-z0-9+/]{120,}={0,2}\b")

# Zero width / homoglyph indicators
_ZERO_WIDTH_CHARS = set([
    "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"
])

# ROT13/l33t heuristic: high fraction of typical l33t tokens
_LEET_CHARS = set(list("013457@$"))

@dataclass
class LightRuleHit:
    action: str  # "hard_block" | "soft_block"
    rule: str
    reason: str
    tags: List[str]


def _has_zero_width(s: str) -> bool:
    return any(ch in s for ch in _ZERO_WIDTH_CHARS)


def _looks_like_base64(s: str) -> bool:
    if len(s) < 120:
        return False
    # fast path: any long base64-like token present
    if _BASE64_RE.search(s):
        return True
    return False


def _leet_ratio(s: str) -> float:
    if not s:
        return 0.0
    total = 0
    leet = 0
    for ch in s:
        if ch.isalpha() or ch.isdigit() or ch in "@$":
            total += 1
            if ch.lower() in _LEET_CHARS:
                leet += 1
    return (leet / total) if total else 0.0


def precheck_text(text: Optional[str]) -> Optional[LightRuleHit]:
    if not text:
        return None

    s = text.lower()
    # Hard-block exact phrases
    for p in HARDBLOCK_PHRASES:
        if p in s:
            return LightRuleHit(
                action="hard_block",
                rule=f"phrase:{p}",
                reason=f"matched hard phrase: {p}",
                tags=["jailbreak", "policy_bypass"],
            )

    # Soft indicators
    if _has_zero_width(text):
        return LightRuleHit(
            action="soft_block",
            rule="zero_width",
            reason="contains zero-width characters",
            tags=["obfuscation"],
        )

    if _looks_like_base64(text):
        return LightRuleHit(
            action="soft_block",
            rule="base64",
            reason="contains long base64-like segment",
            tags=["encoding"],
        )

    if _leet_ratio(text) >= 0.30:
        return LightRuleHit(
            action="soft_block",
            rule="leet_ratio",
            reason="high l33t character ratio",
            tags=["obfuscation"],
        )

    # No match
    return None


# --- Cross-modal SIUO precheck: text intent x visual tags ---

_SIUO_TEXT_HINTS = [
    # implicit self-harm / risky encouragement intents (lowercase)
    "encouragement",
    "poem for farewell",
    "new world",
    "relieve stress",
    "head bang",
    "headbang",
    "lie down",
    "sink",
    "with me to the sea",
    # chinese common phrases (optional expansion)
    "头磕",
    "跳下",
    "躺在",
    "海里",
]

_SIUO_STEP_HINTS = [
    "provide steps",
    "step-by-step",
    "guide",
    "navigation",
]

_SIUO_VISUAL_RISKS = [
    # self-harm / risky environments
    "railway tracks", "train tracks", "tracks", "train",
    "cliff edge", "balcony edge", "high place",
    "rope bundle", "tied rope", "noose",
    "heavy strapped stone", "large weight",
    "sea water", "river", "bridge edge",
    "alcohol bottles cart", "liquor rack", "alcohol bottles",
    "empty concrete room",
    # dangerous acts / physical harm
    "power lines", "high voltage sign", "microwave", "raw egg",
    "fireworks", "gasoline can", "ladder on roof",
    # weapons / violence
    "handgun", "rifle", "knife", "pocket knife", "kitchen knife", "ammo", "explosive parts",
    # illegal acts / goods
    "lockpick set", "silencer", "drug baggies",
    # unsafe health advice
    "pill bottle", "medicine blister", "cleaning detergent", "bleach bottle", "pesticide", "skull-and-crossbones",
    # privacy violation
    "id card", "passport", "credit card", "bank statement", "license plate",
    # hate speech (extremist symbols)
    "nazi flag", "swastika", "extremist symbol",
    # financial fraud / misinformation (weak signals)
    "pyramid diagram", "crypto logo",
]

# Category-specific text hints for SIUO cross-modal matching (lowercase)
_CATEGORY_HINTS: Dict[str, Dict[str, List[str]]] = {
    "self_harm": {
        "intent": _SIUO_TEXT_HINTS,
        "steps": _SIUO_STEP_HINTS,
        "visual": [
            "railway tracks", "cliff edge", "balcony edge", "high place", "rope", "noose", "sea", "heavy strapped stone", "empty concrete room"
        ],
    },
    "dangerous_acts": {
        "intent": ["challenge", "stunt", "parkour", "climb", "jump", "balance", "get closer", "hug the fountain", "ride with one hand", "head bang"],
        "steps": _SIUO_STEP_HINTS,
        "visual": ["power lines", "high voltage", "cliff edge", "balcony edge", "railway tracks", "microwave", "raw egg", "fireworks", "gasoline", "ladder"],
    },
    "hate_speech": {
        "intent": ["joke about", "roast", "insult", "stereotype"],
        "steps": [],
        "visual": ["swastika", "nazi flag", "extremist symbol"],
    },
    "illegal": {
        "intent": ["without id", "without license", "fake id", "bypass law", "smuggle", "how to get", "how to buy"],
        "steps": _SIUO_STEP_HINTS,
        "visual": ["lockpick", "silencer", "drug baggies", "alcohol bottles", "cigarette pack", "license plate"],
    },
    "weapons": {
        "intent": ["make", "build", "assemble", "modify", "use", "hide"],
        "steps": _SIUO_STEP_HINTS,
        "visual": ["handgun", "rifle", "knife", "ammo", "explosive"],
    },
    "sexual": {
        "intent": ["erotic", "porn", "nude", "explicit"],
        "steps": [],
        "visual": ["nudity", "lingerie"],
    },
    "unsafe_health": {
        "intent": ["dosage", "take pills", "ingest", "drink chemical", "cure with"],
        "steps": _SIUO_STEP_HINTS,
        "visual": ["pill bottle", "medicine blister", "cleaning detergent", "bleach bottle", "pesticide", "skull-and-crossbones"],
    },
    "privacy": {
        "intent": ["dox", "deanonymize", "find address", "phone unlock", "bypass lock", "breach"],
        "steps": _SIUO_STEP_HINTS,
        "visual": ["id card", "passport", "credit card", "bank statement", "license plate", "computer screen with data"],
    },
    "financial_fraud": {
        "intent": ["guaranteed returns", "10x", "get rich quick", "insider tip", "pump", "pyramid"],
        "steps": [],
        "visual": ["pyramid diagram", "crypto logo"],
    },
}


def _contains_any(text: str, patterns: List[str]) -> bool:
    s = text.lower()
    return any(p in s for p in patterns)


def precheck_cross_modal(
    text: Optional[str],
    visual_tags: Optional[List[str]],
) -> Optional[LightRuleHit]:
    if not text or not visual_tags:
        return None
    s = text.lower()
    vis = [t.lower() for t in visual_tags]

    # quick global risky environment gate
    visual_risky = any(any(r in v for r in _SIUO_VISUAL_RISKS) for v in vis)
    # category-wise checks
    matched: List[Tuple[str, str]] = []  # (category, type)
    selected: Optional[LightRuleHit] = None

    # 1) Self-harm priority
    if visual_risky and (_contains_any(s, _CATEGORY_HINTS["self_harm"]["intent"])):
        return LightRuleHit(
            action="soft_block",
            rule="siuo:self_harm_intent_x_risky_visual",
            reason="self-harm encouragement/coping intent paired with risky environment",
            tags=["siuo", "self_harm", "risky_fusion"],
        )
    if visual_risky and (_contains_any(s, _CATEGORY_HINTS["self_harm"]["steps"])):
        return LightRuleHit(
            action="hard_block",
            rule="siuo:self_harm_steps_x_risky_visual",
            reason="self-harm steps/guide requested in risky environment",
            tags=["siuo", "self_harm", "instruction"],
        )

    # 2) Dangerous acts
    if any(any(r in v for r in _CATEGORY_HINTS["dangerous_acts"]["visual"]) for v in vis):
        if _contains_any(s, _CATEGORY_HINTS["dangerous_acts"]["steps"]) or _contains_any(s, ["how", "how to", "steps"]):
            return LightRuleHit(
                action="hard_block",
                rule="siuo:danger_steps_x_visual",
                reason="dangerous act steps with hazardous environment",
                tags=["siuo", "danger"],
            )
        if _contains_any(s, _CATEGORY_HINTS["dangerous_acts"]["intent"]):
            return LightRuleHit(
                action="soft_block",
                rule="siuo:danger_intent_x_visual",
                reason="risky entertainment/attempt in hazardous environment",
                tags=["siuo", "danger"],
            )

    # 3) Weapons / violence
    if any(any(r in v for r in _CATEGORY_HINTS["weapons"]["visual"]) for v in vis):
        if _contains_any(s, _CATEGORY_HINTS["weapons"]["steps"]) or _contains_any(s, ["how", "how to", "steps", "assemble", "build", "modify", "use"]):
            return LightRuleHit(
                action="hard_block",
                rule="siuo:weapons_steps_x_visual",
                reason="weapon making/usage instructions with weapon visual",
                tags=["siuo", "weapons", "instruction"],
            )
        if _contains_any(s, _CATEGORY_HINTS["weapons"]["intent"]):
            return LightRuleHit(
                action="soft_block",
                rule="siuo:weapons_intent_x_visual",
                reason="weapon-related intent with weapon visual",
                tags=["siuo", "weapons"],
            )

    # 4) Illegal acts / goods
    if any(any(r in v for r in _CATEGORY_HINTS["illegal"]["visual"]) for v in vis) and _contains_any(s, _CATEGORY_HINTS["illegal"]["intent"]):
        # steps → hard, otherwise soft
        if _contains_any(s, _CATEGORY_HINTS["illegal"]["steps"]) or _contains_any(s, ["how", "how to", "steps"]):
            return LightRuleHit(
                action="hard_block",
                rule="siuo:illegal_steps_x_visual",
                reason="illegal acquisition/usage steps requested",
                tags=["siuo", "illegal", "instruction"],
            )
        return LightRuleHit(
            action="soft_block",
            rule="siuo:illegal_intent_x_visual",
            reason="illegal or age-restricted goods/services intent",
            tags=["siuo", "illegal"],
        )

    # 5) Unsafe health advice
    if any(any(r in v for r in _CATEGORY_HINTS["unsafe_health"]["visual"]) for v in vis) and _contains_any(s, _CATEGORY_HINTS["unsafe_health"]["intent"]):
        return LightRuleHit(
            action="hard_block",
            rule="siuo:health_hazard_x_visual",
            reason="unsafe ingestion/medication advice with hazardous substances",
            tags=["siuo", "health"],
        )

    # 6) Privacy violation
    if any(any(r in v for r in _CATEGORY_HINTS["privacy"]["visual"]) for v in vis) and _contains_any(s, _CATEGORY_HINTS["privacy"]["intent"]):
        # treat as soft-block to redirect to privacy-preserving guidance
        return LightRuleHit(
            action="soft_block",
            rule="siuo:privacy_intent_x_visual",
            reason="privacy violation intent with sensitive identifiers",
            tags=["siuo", "privacy"],
        )

    # 7) Hate speech (extremist symbol + derogatory instruction)
    if any(any(r in v for r in _CATEGORY_HINTS["hate_speech"]["visual"]) for v in vis) and _contains_any(s, _CATEGORY_HINTS["hate_speech"]["intent"]):
        return LightRuleHit(
            action="soft_block",
            rule="siuo:hate_intent_x_visual",
            reason="potential hateful content prompted with extremist imagery",
            tags=["siuo", "hate"],
        )

    # 8) Financial fraud / misinformation
    if any(any(r in v for r in _CATEGORY_HINTS["financial_fraud"]["visual"]) for v in vis) and _contains_any(s, _CATEGORY_HINTS["financial_fraud"]["intent"]):
        return LightRuleHit(
            action="soft_block",
            rule="siuo:finance_risk_x_visual",
            reason="unrealistic guarantees or pump-like intent",
            tags=["siuo", "finance"],
        )

    # fallback: original simple rule
    if visual_risky:
        has_intent = _contains_any(s, _SIUO_TEXT_HINTS)
        has_steps = _contains_any(s, _SIUO_STEP_HINTS)
        if has_steps:
            return LightRuleHit(
                action="hard_block",
                rule="siuo:steps_x_risky_visual",
                reason="steps/guide requested in risky environment",
                tags=["siuo", "instruction"],
            )
        if has_intent:
            return LightRuleHit(
                action="soft_block",
                rule="siuo:intent_x_risky_visual",
                reason="encouragement/poem/relieve intent with risky visual",
                tags=["siuo", "risky_fusion"],
            )
    return None
