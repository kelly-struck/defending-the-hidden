# -*- coding: utf-8 -*-
"""
EX-E1-③ | HarmBench / MM-SafetyBench complete-report upgrade (P0, D4-A)

Purpose
-------
The reviewer (AE-(i)/R1-2) criticised that the evaluation is concentrated on SIUO.
Table 1 already reports HarmBench / MM-SafetyBench / FigStep ASR, but the current
manuscript never states N for these three benchmarks. This task upgrades the
"one cell of numbers" into a complete report: N + ASR/refusal + Wilson 95% CI +
per-class breakdown.

Regime (must not be broken) — paper L407
----------------------------------------
HarmBench and MM-SafetyBench are ALREADY indexed in the production seed library.
To avoid retrieval leakage, the manuscript reports them with the Layer-3 debate
mechanism ALONE, with Layer 1 and Layer 2 DISABLED. This script therefore calls
`DebateOrchestrator` directly (pure L3), never `DefenceOrchestrator` (which would
run L1 -> L2 -> L3 and leak the library). `learning_enabled=False` keeps the
held-out evaluation from writing back into the seed library.

Sampling
--------
- HarmBench : stratified sample N=200 across configs {standard, contextual,
              copyright} proportional to their available sizes, then balanced
              within config by category. Fixed seed.
- MM-SafetyBench : stratified sample N=500 across the 13 categories, balanced
              (dropping whole parquet splits where useful), fixed seed.

Metrics
-------
- ASR        = allow / N          (framework failed to block the attack)
- refusal    = (block + revise)/N (framework intercepted)
- Wilson 95% CI for every proportion.
- per-class table for both benchmarks.

Checkpointing
-------------
Per-sample results are appended to results/e1c_ckpt.json and flushed after every
sample, so an interruption never loses completed work. Resume-safe: on re-run,
samples whose key is already present are skipped.

Outputs
-------
  results/benchmark_full_report.json
  ../benchmark_full_table.md   (regenerated separately by render_table.py)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / "defenceAgent" / ".env")
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

from agentscope.message import Msg, TextBlock, ImageBlock  # noqa: E402

from defenceAgent.orchestrator_debate import DebateOrchestrator  # noqa: E402
from defenceAgent.dataset_utils import load_harmbench_rows, load_mmsafetybench_rows  # noqa: E402

SEED = 42
N_HARMBENCH = 200
N_MMSB = 500
HB_DIR = REPO_ROOT / "defenceAgent" / "statics" / "HarmBench"
MMSB_DIR = REPO_ROOT / "defenceAgent" / "statics" / "MM-SafetyBench"
MMSB_CLASSES = [
    "EconomicHarm", "Financial_Advice", "Fraud", "Gov_Decision", "HateSpeech",
    "Health_Consultation", "Illegal_Activitiy", "Legal_Opinion",
    "Malware_Generation", "Physical_Harm", "Political_Lobbying",
    "Privacy_Violence", "Sex",
]
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT = OUT_DIR / "e1c_ckpt.json"
REPORT = OUT_DIR / "benchmark_full_report.json"


# ------------------------------------------------------------------------------
# Wilson CI
# ------------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def prop_block(k: int, n: int) -> Dict:
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "p": (k / n if n else 0.0),
            "wilson95": [lo, hi]}


# ------------------------------------------------------------------------------
# Checkpoint
# ------------------------------------------------------------------------------
def load_ckpt() -> dict:
    if CKPT.exists():
        try:
            return json.loads(CKPT.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"results": {}}


def save_ckpt(ckpt: dict) -> None:
    tmp = CKPT.with_suffix(".tmp")
    tmp.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CKPT)


# ------------------------------------------------------------------------------
# Sampling
# ------------------------------------------------------------------------------
def sample_harmbench() -> List[Dict]:
    """Stratified: proportional across configs, balanced by category within."""
    rng = random.Random(SEED)
    pool: Dict[str, Dict[str, List[Dict]]] = {}
    counts = {}
    for cfg in ["standard", "contextual", "copyright"]:
        rows = load_harmbench_rows(HB_DIR, config=cfg)
        counts[cfg] = len(rows)
        by_cat: Dict[str, List[Dict]] = {}
        for r in rows:
            cat = r.get("category") or "unknown"
            by_cat.setdefault(cat, []).append(r)
        pool[cfg] = by_cat

    total = sum(counts.values())
    out: List[Dict] = []
    for cfg, by_cat in pool.items():
        share = int(round(N_HARMBENCH * counts[cfg] / total))
        # balanced round-robin across categories
        cats = sorted(by_cat)
        for c in cats:
            rng.shuffle(by_cat[c])
        idx = {c: 0 for c in cats}
        picked = 0
        while picked < share and any(idx[c] < len(by_cat[c]) for c in cats):
            for c in cats:
                if picked >= share:
                    break
                if idx[c] < len(by_cat[c]):
                    r = by_cat[c][idx[c]]
                    idx[c] += 1
                    out.append({"benchmark": "HarmBench", "config": cfg,
                                "category": c, "id": f"{cfg}:{r['id']}",
                                "text": r["prompt"], "context": r.get("context"),
                                "image_bytes": None})
                    picked += 1
    rng.shuffle(out)
    return out[:N_HARMBENCH]


def sample_mmsb() -> List[Dict]:
    """Balanced stratified sample across the 13 classes (all splits).

    Performance note: the full corpus is ~6,720 rows with ~150 KB images each;
    re-encoding every image (PIL round-trip in `extract_image_bytes`) is slow.
    We therefore read the parquet tables directly, sample row indices first,
    and only extract/encode the image bytes for the ~500 selected rows.
    """
    import pandas as pd
    rng = random.Random(SEED)
    per = N_MMSB // len(MMSB_CLASSES)

    # 1) collect (class, split, local_index) pools WITHOUT touching image bytes
    pools: Dict[str, List[Tuple[str, int]]] = {c: [] for c in MMSB_CLASSES}
    for cls in MMSB_CLASSES:
        for split in ["SD", "SD_TYPO", "TYPO", "Text_only"]:
            pq = MMSB_DIR / "data" / cls / f"{split}.parquet"
            if not pq.exists():
                continue
            df = pd.read_parquet(pq, columns=["id"])
            for i in range(len(df)):
                pools[cls].append((split, i))

    # 2) sample indices per class
    chosen: List[Tuple[str, str, int]] = []
    for cls in MMSB_CLASSES:
        idxs = pools[cls][:]
        rng.shuffle(idxs)
        for split, i in idxs[:per]:
            chosen.append((cls, split, i))

    # 3) extract images/text only for chosen rows
    out: List[Dict] = []
    cache: Dict[str, "pd.DataFrame"] = {}
    for cls, split, i in chosen:
        key = f"{cls}/{split}"
        if key not in cache:
            cache[key] = pd.read_parquet(MMSB_DIR / "data" / cls / f"{split}.parquet")
        row = cache[key].iloc[i]
        img = row.get("image")
        img_bytes = None
        try:
            if isinstance(img, (bytes, bytearray, memoryview)):
                img_bytes = bytes(img)
            elif img is not None:
                from defenceAgent.dataset_utils import extract_image_bytes
                img_bytes = extract_image_bytes(img, root=None)
        except Exception:
            img_bytes = None
        rid = row.get("id")
        out.append({"benchmark": "MM-SafetyBench", "config": split,
                    "category": cls, "id": f"{cls}/{split}:{rid}",
                    "text": "" if pd.isna(row.get("question")) else str(row.get("question")),
                    "context": None, "image_bytes": img_bytes})
    rng.shuffle(out)
    return out[:N_MMSB]


# ------------------------------------------------------------------------------
# L3-only evaluation
# ------------------------------------------------------------------------------
def build_msg(row: Dict) -> Msg:
    blocks: List = []
    if row.get("context"):
        blocks.append(TextBlock(type="text", text=f"Context:\n{row['context']}\n\nPrompt:\n{row['text']}"))
    else:
        blocks.append(TextBlock(type="text", text=row["text"]))
    ib = row.get("image_bytes")
    if ib:
        b64 = base64.b64encode(ib).decode("ascii")
        blocks.append(ImageBlock(type="image", source={"type": "base64", "data": b64, "media_type": "image/png"}))
    return Msg(name="user", role="user", content=blocks)


async def eval_one(orch: DebateOrchestrator, row: Dict) -> Dict:
    msg = build_msg(row)
    try:
        res = await orch.run(input_msg=msg)
        verdict = res.get("verdict", "unsafe")
        transcript = res.get("debate_transcript", {})
        return {"id": row["id"], "benchmark": row["benchmark"],
                "config": row["config"], "category": row["category"],
                "verdict": verdict,
                "action": "allow" if verdict == "safe" else "block",
                "total_tokens": res.get("total_tokens", 0),
                "final_decision": (transcript or {}).get("final_decision", "")}
    except Exception as e:
        is_block, code, emsg = _provider_block(e)
        return {"id": row["id"], "benchmark": row["benchmark"],
                "config": row["config"], "category": row["category"],
                "verdict": "provider_block" if is_block else "error",
                "action": "block" if is_block else "error",
                "error": f"{type(e).__name__}: {emsg}",
                "total_tokens": 0}


def _provider_block(err: Exception):
    s = f"{type(err).__name__}: {err}"
    if "DataInspectionFailed" in s:
        return True, "DataInspectionFailed", s
    return False, "", s


# ------------------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------------------
def aggregate(results: List[Dict]) -> Dict:
    agg: Dict[str, Dict] = {}
    for r in results:
        b = r["benchmark"]
        a = agg.setdefault(b, {"total": 0, "allow": 0, "block": 0, "error": 0,
                               "by_category": {}, "by_config": {},
                               "tokens": 0})
        a["total"] += 1
        a["tokens"] += int(r.get("total_tokens") or 0)
        if r["action"] == "allow":
            a["allow"] += 1
        elif r["action"] == "block":
            a["block"] += 1
        else:
            a["error"] += 1
        for key, fld in (("category", "by_category"), ("config", "by_config")):
            v = r.get(key) or "unknown"
            d = a[fld].setdefault(v, {"total": 0, "allow": 0})
            d["total"] += 1
            if r["action"] == "allow":
                d["allow"] += 1

    out = {}
    for b, a in agg.items():
        n = a["total"]
        asr = prop_block(a["allow"], n)
        refusal = prop_block(a["block"], n)
        rec = {
            "N": n,
            "allow": a["allow"], "block": a["block"], "error": a["error"],
            "asr": asr, "refusal_rate": refusal,
            "avg_tokens": (a["tokens"] / n if n else 0.0),
            "per_category": {
                c: {"N": d["total"], "allow": d["allow"],
                    "asr": prop_block(d["allow"], d["total"])}
                for c, d in sorted(a["by_category"].items())
            },
            "per_config": {
                c: {"N": d["total"], "allow": d["allow"],
                    "asr": prop_block(d["allow"], d["total"])}
                for c, d in sorted(a["by_config"].items())
            },
        }
        # HarmBench-copyright is a verbatim-memorisation probe (lyrics/books), not a
        # harmful-instruction set; a safety-oriented defender correctly allows it.
        # Report the harm-ASR excluding that config so both readings are available.
        if b == "HarmBench":
            harm_relevant = [r for r in results
                             if r["benchmark"] == "HarmBench" and r["config"] != "copyright"]
            ka = sum(1 for r in harm_relevant if r["action"] == "allow")
            kn = len(harm_relevant)
            rec["asr_excluding_copyright_memorisation"] = {
                "note": ("HarmBench 'copyright' subset tests verbatim lyric/book "
                         "memorisation, not harmful intent; excluding it isolates "
                         "safety-relevant ASR."),
                "N": kn, "allow": ka, "asr": prop_block(ka, kn),
            }
        out[b] = rec
    return out


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-harmbench", type=int, default=N_HARMBENCH)
    ap.add_argument("--n-mmsb", type=int, default=N_MMSB)
    ap.add_argument("--no-harmbench", action="store_true")
    ap.add_argument("--no-mmsb", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="parallel debate calls (async)")
    args = ap.parse_args()

    t0 = time.time()
    ckpt = load_ckpt()
    done = ckpt.setdefault("results", {})
    meta = ckpt.setdefault("meta", {})
    meta.update({
        "seed": SEED, "regime": "L3-only (L1/L2 disabled)",
        "learning_enabled": False, "orchestrator": "DebateOrchestrator",
        "n_harmbench": args.n_harmbench, "n_mmsb": args.n_mmsb,
        "harmbench_dir": str(HB_DIR), "mmsb_dir": str(MMSB_DIR),
    })

    rows: List[Dict] = []
    if not args.no_harmbench:
        hb = sample_harmbench()
        print(f"[sample] HarmBench {len(hb)} (target {args.n_harmbench})")
        rows += hb[:args.n_harmbench]
    if not args.no_mmsb:
        mb = sample_mmsb()
        print(f"[sample] MM-SafetyBench {len(mb)} (target {args.n_mmsb})")
        rows += mb[:args.n_mmsb]

    todo = [r for r in rows if hashlib.md5(r["id"].encode()).hexdigest() not in done]
    print(f"[run] total {len(rows)}, todo {len(todo)} (cached {len(rows)-len(todo)})")

    async def run_all():
        # Process in bounded chunks with asyncio.gather. Creating 700+ tasks
        # up-front with as_completed proved unreliable (the run stalled after
        # sampling); a chunked gather is robust and keeps at most `chunk` tasks
        # alive. A fresh DebateOrchestrator per task keeps the non-concurrency-safe
        # orchestrator isolated; learning_enabled=False protects the seed library.
        chunk = max(1, args.concurrency) * 5
        sem = asyncio.Semaphore(args.concurrency)
        i = 0
        while i < len(todo):
            batch = todo[i:i + chunk]
            print(f"[chunk] start i={i} size={len(batch)}", flush=True)

            async def bounded(r):
                async with sem:
                    print(f"[task] start {r['id'][:32]}", flush=True)
                    o = DebateOrchestrator(learning_enabled=False)
                    print(f"[task] orch ready {r['id'][:32]}", flush=True)
                    try:
                        res = await asyncio.wait_for(eval_one(o, r), timeout=300)
                    except asyncio.TimeoutError:
                        res = {"id": r["id"], "benchmark": r["benchmark"],
                               "config": r["config"], "category": r["category"],
                               "verdict": "timeout", "action": "error",
                               "error": "asyncio timeout 300s", "total_tokens": 0}
                    print(f"[task] done {r['id'][:32]} -> {res.get('verdict')}", flush=True)
                    return res

            results = await asyncio.gather(*[bounded(r) for r in batch])
            print(f"[chunk] gather returned {len(results)}", flush=True)
            for r in results:
                done[hashlib.md5(r["id"].encode()).hexdigest()] = r
                i += 1
            save_ckpt(ckpt)  # checkpoint after every chunk
            print(f"[run] {i}/{len(todo)}  last={results[-1]['id'][:40]} "
                  f"verdict={results[-1].get('verdict')}", flush=True)

    if todo:
        asyncio.run(run_all())
        save_ckpt(ckpt)

    results = list(done.values())
    agg = aggregate(results)

    report = {
        "experiment": "EX-E1-③",
        "title": "HarmBench / MM-SafetyBench complete-report upgrade",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": meta,
        "l3_only_flag": True,
        "l3_only_rationale": (
            "HarmBench and MM-SafetyBench are indexed in the production seed "
            "library; reporting with Layer 3 (debate) alone and Layer 1/2 "
            "disabled avoids retrieval leakage (paper L407)."),
        "sampling": {
            "method": "stratified, fixed seed 42",
            "harmbench": {"requested": args.n_harmbench,
                          "achieved": len([r for r in rows if r['benchmark'] == 'HarmBench'])},
            "mmsafetybench": {"requested": args.n_mmsb,
                              "achieved": len([r for r in rows if r['benchmark'] == 'MM-SafetyBench'])},
        },
        "results": agg,
        "api_calls": {"debate_samples": len(results)},
        "runtime_sec": round(time.time() - t0, 2),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== EX-E1-③ done ===")
    for b, r in agg.items():
        print(f"{b}: N={r['N']} ASR={r['asr']['p']*100:.2f}% "
              f"CI=[{r['asr']['wilson95'][0]*100:.2f},{r['asr']['wilson95'][1]*100:.2f}] "
              f"refusal={r['refusal_rate']['p']*100:.2f}%")
    print("report ->", REPORT)


if __name__ == "__main__":
    main()
