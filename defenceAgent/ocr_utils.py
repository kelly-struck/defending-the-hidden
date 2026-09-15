from __future__ import annotations
"""Best-effort OCR utilities for lightweight image-side rule precheck.

Design goals:
- Optional dependency: uses pytesseract if available, otherwise no-op.
- No network access: only handles in-message base64 images.
- Keep it fast: minimal preprocessing; concatenate OCR from up to a few images.
"""

from typing import Optional, List
from io import BytesIO

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - optional
    pytesseract = None  # type: ignore

from PIL import Image
import base64


def _ocr_image_bytes(data: bytes, lang: Optional[str] = None) -> str:
    if pytesseract is None:
        return ""
    try:
        with Image.open(BytesIO(data)) as im:
            im = im.convert("L")  # grayscale
            # light binarization can help text visibility
            try:
                import numpy as np  # type: ignore

                arr = np.array(im)
                # Otsu-like quick threshold heuristic
                thr = 180
                arr = (arr > thr) * 255
                im = Image.fromarray(arr.astype('uint8'), mode='L')
            except Exception:
                pass
            cfg = {}
            if lang:
                cfg["lang"] = lang
            text = pytesseract.image_to_string(im, **cfg).strip()
            return text
    except Exception:
        return ""


def _decode_base64_to_bytes(b64: str) -> Optional[bytes]:
    try:
        return base64.b64decode(b64, validate=False)
    except Exception:
        try:
            return base64.b64decode(b64 + "==")
        except Exception:
            return None


def ocr_text_from_msg_content_blocks(blocks: List[dict], max_images: int = 3) -> Optional[str]:
    if not blocks:
        return None
    texts: List[str] = []
    count = 0
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") != "image":
            continue
        src = b.get("source")
        if not isinstance(src, dict):
            continue
        if src.get("type") == "base64":
            data = src.get("data")
            if isinstance(data, str):
                raw = _decode_base64_to_bytes(data)
                if raw:
                    txt = _ocr_image_bytes(raw)
                    if txt:
                        texts.append(txt)
                        count += 1
                        if count >= max_images:
                            break
        # URL images are intentionally skipped to avoid network I/O
    all_text = "\n".join([t for t in texts if t.strip()])
    return all_text if all_text.strip() else None
