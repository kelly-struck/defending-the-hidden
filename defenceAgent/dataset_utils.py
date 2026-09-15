from __future__ import annotations
"""Utilities for loading MM-SafetyBench parquet and extracting images.

This module helps both eval_compare.py and eval_figstep.py to use the same
dataset rows and provides robust image-bytes extraction for various parquet
encodings (path, bytes, dict-like, PIL Image, etc.).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import io

import pandas as pd
from PIL import Image


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _bytes_to_png_bytes(raw: bytes) -> bytes:
    """Attempt to normalize arbitrary image bytes to PNG bytes."""
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            return _to_png_bytes(im)
    except Exception:
        # fallback: return as-is; downstream may still accept
        return raw


def extract_image_bytes(v: Any, root: Optional[Path] = None) -> Optional[bytes]:
    """Extract image bytes from a parquet cell value.

    Supports common encodings:
    - str path (absolute or relative to root)
    - bytes
    - dict with keys like {"path": str} or {"bytes": bytes}
    - PIL.Image.Image
    Returns PNG bytes when possible.
    """
    if v is None:
        return None

    # case: PIL Image
    if isinstance(v, Image.Image):
        return _to_png_bytes(v.convert("RGB"))

    # case: bytes
    if isinstance(v, (bytes, bytearray, memoryview)):
        return _bytes_to_png_bytes(bytes(v))

    # case: path string
    if isinstance(v, str):
        p = Path(v)
        if not p.is_file() and root is not None:
            p = (root / v).resolve()
        try:
            data = p.read_bytes()
            return _bytes_to_png_bytes(data)
        except Exception:
            return None

    # case: dict-like
    if isinstance(v, dict):
        # huggingface image cell may have 'path' or 'bytes'
        if "path" in v and isinstance(v["path"], str):
            return extract_image_bytes(v["path"], root=root)
        if "bytes" in v and isinstance(v["bytes"], (bytes, bytearray, memoryview)):
            return _bytes_to_png_bytes(bytes(v["bytes"]))
        # some variants
        if "data" in v and isinstance(v["data"], (bytes, bytearray, memoryview)):
            return _bytes_to_png_bytes(bytes(v["data"]))

    # unsupported
    return None


def load_mmsafetybench_rows(
    dataset_dir: str | Path,
    branch: str,
    split: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load rows from local parquet (MM-SafetyBench clone) and return a list of
    dicts: {"id": str, "question": str, "image_bytes": Optional[bytes]}.
    """
    dataset_dir = Path(dataset_dir)
    parquet_path = dataset_dir / "data" / branch / f"{split}.parquet"
    df = pd.read_parquet(parquet_path)
    if limit is not None:
        df = df.head(limit)

    rows: List[Dict[str, Any]] = []
    root_for_paths = parquet_path.parent
    for _, r in df.iterrows():
        q = r.get("question")
        img_cell = r.get("image")
        img_bytes = extract_image_bytes(img_cell, root=root_for_paths)
        rid = r.get("id")
        rows.append({
            "id": None if pd.isna(rid) else str(rid),
            "question": "" if pd.isna(q) else str(q),
            "image_bytes": img_bytes,
        })
    return rows


def make_image_block_base64(img_bytes: bytes) -> Dict[str, Any]:
    import base64
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return {"type": "base64", "data": b64, "media_type": "image/png"}


def load_mmvet_rows(
    dataset_dir: str | Path,
    split_glob: str = "test-*.parquet",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load rows from local MM-Vet parquet (HF offline clone) and return list of dicts.

    Expected local layout (already present in this repo under defenceAgent/statics/MM-Vet):
      <dataset_dir>/data/test-00000-of-00001.parquet

    Schema (see README.md in the dataset folder):
      - id: string
      - image: image (path/bytes/dict)
      - question: string
      - answer: string (may contain <AND>/<OR> tokens for composite matching)
      - capability: List[string]
    """
    dataset_dir = Path(dataset_dir)
    data_dir = dataset_dir / "data"
    # find first parquet by glob, default to test-*
    files = sorted(data_dir.glob(split_glob))
    if not files:
        raise FileNotFoundError(f"未在 {data_dir} 下找到 parquet（模式 {split_glob}）")
    parquet_path = files[0]

    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if limit is not None:
        df = df.head(limit)

    rows: List[Dict[str, Any]] = []
    root_for_paths = parquet_path.parent
    for _, r in df.iterrows():
        rid = r.get("id")
        q = r.get("question")
        gt = r.get("answer")
        caps = r.get("capability")
        img_cell = r.get("image")
        img_bytes = extract_image_bytes(img_cell, root=root_for_paths)
        # capability 字段可能是 list/tuple/np.ndarray/单值，统一为 list[str]
        cap_list: List[str]
        caps_val = caps
        try:
            import numpy as _np  # type: ignore
        except Exception:
            _np = None  # noqa: N816

        if caps_val is None:
            cap_list = []
        elif isinstance(caps_val, (list, tuple)):
            cap_list = [str(x) for x in caps_val]
        elif (_np is not None) and isinstance(caps_val, _np.ndarray):  # type: ignore[attr-defined]
            cap_list = [str(x) for x in caps_val.tolist()]
        else:
            cap_list = [str(caps_val)]

        rows.append({
            "id": None if pd.isna(rid) else str(rid),
            "question": "" if pd.isna(q) else str(q),
            "answer": None if pd.isna(gt) else str(gt),
            "capability": cap_list,
            "image_bytes": img_bytes,
        })
    return rows


def load_harmbench_rows(
    dataset_dir: str | Path,
    config: str = "standard",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load text-only jailbreak prompts from HarmBench (local clone).

    Local expected layout (already present under defenceAgent/statics/HarmBench):
      <dataset_dir>/<config>/train-00000-of-00001.parquet

    Configs:
      - standard: columns [prompt, category]
      - contextual: columns [prompt, context, category]
      - copyright: columns [prompt, tags]

    Returns a list of dicts:
      {"id": str, "prompt": str, "context": Optional[str], "category": Optional[str]}
    """
    dataset_dir = Path(dataset_dir)
    subdir = dataset_dir / config
    files = sorted(subdir.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"未在 {subdir} 下找到 parquet（模式 train-*.parquet）")
    parquet_path = files[0]

    df = pd.read_parquet(parquet_path)
    if limit is not None and limit > 0:
        df = df.head(limit)

    rows: List[Dict[str, Any]] = []
    for idx, r in df.iterrows():
        rid = str(idx)
        prompt = r.get("prompt")
        ctx = r.get("context") if "context" in df.columns else None
        cat = None
        if "category" in df.columns:
            cat = r.get("category")
        elif "tags" in df.columns:
            cat = r.get("tags")
        rows.append({
            "id": rid,
            "prompt": "" if pd.isna(prompt) else str(prompt),
            "context": None if ctx is None or pd.isna(ctx) else str(ctx),
            "category": None if cat is None or pd.isna(cat) else str(cat),
        })
    return rows

