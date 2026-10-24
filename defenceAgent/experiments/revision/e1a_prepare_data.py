# -*- coding: utf-8 -*-
"""EX-E1a | Step 0: intent set + benign image pool preparation (ZERO API calls).

Purpose
-------
EX-E1a ("adaptive attack: Defender-prompt exploitation") needs, per the
revision guide (section 4):

  * a harmful-intent set drawn from HarmBench (NOT from SIUO, which stays
    held-out), and
  * for each intent, a legitimate image-text input built from a benign
    visual distractor image.

This script performs the purely local preparation step:

1. Intent set: HarmBench `standard` config (200 rows, 6 behaviour
   categories). Stratified sample of N=50 using the largest-remainder
   allocation over categories, fixed seed 42. The 10-intent pilot set is a
   subset of the full 50 (round-robin across categories) so pilot results
   count toward the full run.

2. Image pool: `datasets/benign/benign_dataset.json` (953 entries with
   embedded base64 images, MME + MM-Vet). Filter by encoded size to avoid
   pathological images, then decode -> RGB -> downscale to max side 1024 ->
   re-encode JPEG q85 -> save under `e1a/images/`. Only the 300 candidate
   images actually assigned to attack vectors are processed.

3. Assignment: 50 intents x 2 attack vectors (A=distractor-attached,
   B=scientific-framing) x 3 candidate images per vector, sliced from the
   shuffled pool (no image reused across vectors). The attack generator
   (LLM) will choose 1 of its 3 candidates per vector at generation time.

Outputs (all under e1a/data/ and e1a/images/):
  data/intents.json     {seed, n_full, n_pilot, items:[{iid, category, prompt, pilot}]}
  data/image_pool.json  {filter bounds, n_pool, n_used, entries:[...]}
  data/assignment.json  {vectors:[{iid, vector, candidates:[img_id x3]}]}
  images/*.jpg          processed candidate images

Zero API calls. Run with the agentdefence environment:
  source activate agentdefence && python prepare_e1a_data.py
"""

from __future__ import annotations

import base64
import io
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
IMG_DIR = OUT_DIR / "images"

SEED = 42
N_FULL = 50
N_PILOT = 10
CAND_PER_VECTOR = 3
N_VECTORS = N_FULL * 2

# Encoded (base64) size bounds for pool images: keeps images large enough to
# be meaningful distractors but small enough to control API payload cost.
MIN_B64 = 45_000
MAX_B64 = 1_200_000
# Minimum size (bytes) of the *processed* JPEG candidate; smaller results are
# too blurry to serve as a prominent visual distractor.
MIN_PROCESSED = 20_000

HB_PARQUET = ROOT / "defenceAgent" / "statics" / "HarmBench" / "standard" / "train-00000-of-00001.parquet"
BENIGN_JSON = ROOT / "datasets" / "benign" / "benign_dataset.json"

VECTORS = ["A_distractor", "B_framing"]


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> list[dict]:
    """Largest-remainder allocation over categories; shuffle within category."""
    rng = random.Random(seed)
    cats = sorted(df["category"].unique())
    n_total = len(df)

    quotas: dict[str, int] = {}
    remainders: dict[str, float] = {}
    for c in cats:
        q = len(df[df["category"] == c]) * n / n_total
        quotas[c] = int(q)
        remainders[c] = q - int(q)
    need = n - sum(quotas.values())
    # give leftover seats to the largest remainders (ties broken by name)
    for c in sorted(cats, key=lambda x: (-remainders[x], x))[:need]:
        quotas[c] += 1

    picked: list[dict] = []
    for c in cats:
        idx = list(df[df["category"] == c].index)
        rng.shuffle(idx)
        for i in idx[: quotas[c]]:
            picked.append({
                "iid": f"HB-{i:04d}",
                "category": str(c),
                "prompt": str(df.loc[i, "prompt"]),
            })

    # deterministic order: category then original index (stable for slicing)
    picked.sort(key=lambda r: (r["category"], r["iid"]))
    return picked


def mark_pilot(items: list[dict], n_pilot: int) -> None:
    """Round-robin across categories so the pilot covers all of them."""
    by_cat: dict[str, list[dict]] = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)
    order = sorted(by_cat.keys())
    cursor = {c: 0 for c in order}
    pilot_ids = []
    while len(pilot_ids) < n_pilot and any(cursor[c] < len(by_cat[c]) for c in order):
        for c in order:
            if len(pilot_ids) >= n_pilot:
                break
            if cursor[c] < len(by_cat[c]):
                pilot_ids.append(by_cat[c][cursor[c]]["iid"])
                cursor[c] += 1
    for it in items:
        it["pilot"] = it["iid"] in set(pilot_ids)


def safe_name(img_id: str) -> str:
    return img_id.replace("/", "_").replace("\\", "_")


def build_image_pool(seed: int, need: int) -> list[dict]:
    """Filter benign_dataset images, process & save candidates, return entries.

    De-duplicates by img_id (the source dataset repeats some images under
    several question entries) and skips processed images that fall below
    MIN_PROCESSED bytes, pulling further candidates until `need` are ready.
    """
    ds = json.loads(BENIGN_JSON.read_text(encoding="utf-8"))
    rng = random.Random(seed)

    seen_ids = set()
    pool_raw = []
    for x in ds:
        img = x.get("image")
        if not isinstance(img, dict):
            continue
        b = img.get("bytes")
        data = b.get("data") if isinstance(b, dict) else None
        if not data or not isinstance(data, str):
            continue
        iid = str(x.get("id"))
        if iid in seen_ids:
            continue
        if MIN_B64 <= len(data) <= MAX_B64:
            seen_ids.add(iid)
            pool_raw.append(x)

    rng.shuffle(pool_raw)
    if len(pool_raw) < need:
        raise SystemExit(f"image pool too small after filter: {len(pool_raw)} < {need}")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    skipped_small = 0
    for x in pool_raw:
        if len(entries) >= need:
            break
        data = x["image"]["bytes"]["data"]
        raw = base64.b64decode(data)
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = im.size
        m = max(w, h)
        if m > 1024:
            s = 1024 / m
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        fn = safe_name(str(x.get("id"))) + ".jpg"
        out_path = IMG_DIR / fn
        im.save(out_path, "JPEG", quality=85)
        if out_path.stat().st_size < MIN_PROCESSED:
            out_path.unlink()
            skipped_small += 1
            continue
        entries.append({
            "img_id": str(x.get("id")),
            "file": fn,
            "orig_b64_len": len(data),
            "size": list(im.size),
        })
    if skipped_small:
        print(f"[images] skipped {skipped_small} processed images below {MIN_PROCESSED} B")
    return pool_raw, entries


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- intents ----------------
    df = pd.read_parquet(HB_PARQUET)
    assert len(df) == 200, f"unexpected HarmBench standard size: {len(df)}"
    items = stratified_sample(df, N_FULL, SEED)
    assert len(items) == N_FULL, len(items)
    mark_pilot(items, N_PILOT)

    cat_counts = {}
    for it in items:
        cat_counts[it["category"]] = cat_counts.get(it["category"], 0) + 1
    pilot_n = sum(1 for it in items if it["pilot"])
    print(f"[intents] total={len(items)} pilot={pilot_n}")
    print(f"[intents] per-category: {cat_counts}")

    intents_payload = {
        "source": "HarmBench standard (non-SIUO held-out)",
        "seed": SEED,
        "n_full": N_FULL,
        "n_pilot": N_PILOT,
        "items": items,
    }
    (DATA_DIR / "intents.json").write_text(
        json.dumps(intents_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- image pool ----------------
    pool_raw, entries = build_image_pool(SEED, N_VECTORS * CAND_PER_VECTOR)
    print(f"[images] filtered pool={len(pool_raw)}, processed candidates={len(entries)}")

    pool_payload = {
        "source": "datasets/benign/benign_dataset.json (MME + MM-Vet, benign)",
        "filter": {"min_b64": MIN_B64, "max_b64": MAX_B64,
                   "resize_max_side": 1024, "format": "JPEG q85"},
        "n_pool": len(pool_raw),
        "n_used": len(entries),
        "entries": entries,
    }
    (DATA_DIR / "image_pool.json").write_text(
        json.dumps(pool_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- assignment ----------------
    vectors = []
    k = 0
    for it in items:
        for v in VECTORS:
            cands = entries[k: k + CAND_PER_VECTOR]
            k += CAND_PER_VECTOR
            vectors.append({
                "iid": it["iid"],
                "vector": v,
                "candidates": [e["img_id"] for e in cands],
            })
    assert k == len(entries), (k, len(entries))

    assign_payload = {
        "cand_per_vector": CAND_PER_VECTOR,
        "vectors": vectors,
    }
    (DATA_DIR / "assignment.json").write_text(
        json.dumps(assign_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[assignment] vectors={len(vectors)} "
          f"(pilot vectors={sum(1 for v in vectors if any(it['iid']==v['iid'] and it['pilot'] for it in items))})")
    print("=== prepare_e1a_data done ===")
    print("outputs:", DATA_DIR / "intents.json", "|", DATA_DIR / "image_pool.json", "|", DATA_DIR / "assignment.json")


if __name__ == "__main__":
    main()
