# -*- coding: utf-8 -*-
"""P0-5 TTL mechanism — redesigned experiment (competence-based retention).

Follows the retention-policy protocol reported in Supplementary Appendix D:

  Step 1  Held-out library construction:
            L_threat  = MM-SafetyBench (SD) threat entries, >=500, disjoint from SIUO
            P_poison  = benign queries whose embedding is near L_threat (constructively retrievable)
            Q_attack  = 167 SIUO probes (disjoint from L_threat)
            Q_benign  = 1000 MME/MM-Vet benign queries
            R_rare    = 20 rare genuine threats injected into the library (Step 6 sub-experiment)
  Step 2  Poison entries are retrievable BY CONSTRUCTION (sim >= tau against some library entry).
  Step 3  Compare three retention policies: no-TTL / current-TTL / competence-TTL.
  Step 4  Simulated time window: 20 steps x 1000 queries (r=0.99 benign), attack-variant
          batches injected every 5 steps (the 20 rare entries appear exactly once each).
  Step 5  Report: poison activation rate, poison recall, collateral removal,
          rare-threat retention, FPR(post), TPR(post).
  Step 6  Rare-threat sub-experiment: survival of 20 rare entries that are activated once.

Retrieval signal: real Qwen2.5-VL-Embedding (multimodal text+image), same model as
production. IMPORTANT empirical finding: under the real qwen2.5-vl-embedding,
cross-modal cosine similarities are far below the paper's tau=0.85 (SIUO-SIUO max 0.704;
SIUO->HarmBench-text max 0.609). We therefore set tau from the measured distribution
(Youden-optimal on attack-vs-benign activation), which is itself part of the honest
finding: the old "0/17 poison activation" experiment never reached the mechanism, and
the nominal tau=0.85 is unreachable with this embedding model.

Usage:
  python run_ttl_experiment.py --phase embed     # embed library+benign+rare, cache to disk
  python run_ttl_experiment.py --phase simulate  # run simulation, write report
  python run_ttl_experiment.py --phase all       # both
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]          # repository root
EXP_DIR = Path(__file__).resolve().parent
CACHE_DIR = EXP_DIR / "embeddings"
OUT_DIR = EXP_DIR / "results"

SIUO_PATH = ROOT / "defenceAgent" / "statics" / "SIUO" / "siuo_gen.json"
MSB_DIR = ROOT / "defenceAgent" / "statics" / "MM-SafetyBench" / "data"
BENIGN_PATH = ROOT / "datasets" / "benign" / "benign_dataset.json"
CHROMA_DIR = ROOT / "chroma_db"
SIUO_COLLECTION = "siuo_poisoning_robustness_v2"

EMBED_MODEL = "qwen2.5-vl-embedding"

# Experiment parameters (Step 4 protocol)
N_STEPS = 20
STEP_SIZE = 1000
R_BENIGN = 0.99
N_BENIGN_PER_STEP = int(STEP_SIZE * R_BENIGN)      # 990
N_ATTACK_PER_STEP = STEP_SIZE - N_BENIGN_PER_STEP  # 10
RARE_BATCH_STEPS = [4, 9, 14, 19]                  # 0-indexed: every 5 steps (5,10,15,20)
RARE_PER_BATCH = 5                                  # 20 rare entries total, once each

# Library composition
N_CLEAN = 500           # L_threat (MM-SafetyBench SD)
N_RARE = 20             # R_rare (held-out MM-SafetyBench, injected into library)
N_POISON = 50           # P_poison (>=50 per guide)
N_ATTACK_MAIN = 167     # Q_attack_main: held-out MM-SafetyBench attack probes (disjoint from L_threat)

# Retention parameters (guide §5.3)
TTL_BASE = 10.0         # current-TTL: fixed lifetime (steps) after last activation
TTL_MIN = 10.0          # competence: initial TTL (kappa=0.5) = 20 >= last rare batch (t=19)
TTL_MAX = 30.0
DELTA = 0.5             # competence: R <- delta*R on each refutation
QUARANTINE_O = 3        # competence: o_i >= 3 -> quarantine
ALPHA = 1.0
BETA = 1.0

SEED = 42

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[ttl-exp] {msg}", flush=True)


def _img_bytes_to_b64(img_bytes: bytes, max_side: int = 512, quality: int = 78) -> str:
    """JPEG bytes -> compressed base64 data URI (keeps API payload small)."""
    from PIL import Image
    buf = io.BytesIO(img_bytes)
    try:
        img = Image.open(buf).convert("RGB")
    except Exception:
        img = Image.new("RGB", (32, 32), (128, 128, 128))
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_siuo_questions() -> List[str]:
    data = json.loads(SIUO_PATH.read_text(encoding="utf-8"))
    return [x["question"].strip() for x in data]


def load_siuo_embeddings_from_chroma() -> np.ndarray:
    """Reuse the cached 167 SIUO multimodal embeddings (verified 1:1 with siuo_gen.json)."""
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(SIUO_COLLECTION)
    data = col.get(include=["embeddings", "documents"])
    docs = data["documents"]
    embs = np.asarray(data["embeddings"], dtype=np.float64)
    # align to siuo_gen.json order
    q_order = load_siuo_questions()
    texts = []
    for d in docs:
        try:
            texts.append(json.loads(d).get("text", "").strip())
        except Exception:
            texts.append("")
    idx = []
    qset = {q: i for i, q in enumerate(q_order)}
    for t in texts:
        if t in qset:
            idx.append(qset[t])
        else:
            raise ValueError(f"SIUO chroma doc text not found in siuo_gen.json: {t[:80]}")
    if len(idx) != len(q_order):
        raise ValueError("SIUO chroma cache does not cover all 167 samples")
    return embs[np.argsort(idx)]


def load_mmsb_items(n_clean: int, n_rare: int, seed: int = SEED) -> Dict[str, Any]:
    """Load (n_clean + n_rare) MM-SafetyBench SD (image+text) threat items, held out.
    Each item: {qid, text, image_b64}. Sampling is class-balanced and seed-fixed."""
    import pandas as pd
    subs = sorted(p for p in os.listdir(MSB_DIR)
                  if os.path.isdir(MSB_DIR / p) and (MSB_DIR / p / "SD.parquet").exists())
    items: List[Dict[str, Any]] = []
    for sub in subs:
        df = pd.read_parquet(MSB_DIR / sub / "SD.parquet")
        for _, row in df.iterrows():
            items.append({
                "text": str(row["question"]).strip(),
                "image_bytes": bytes(row["image"]),
            })
    rng = random.Random(seed)
    rng.shuffle(items)
    clean, rare = items[:n_clean], items[n_clean:n_clean + n_rare]
    log(f"MM-SafetyBench: {len(items)} SD items; clean={len(clean)}, rare={len(rare)}")
    return {"clean": clean, "rare": rare}


def load_benign_items() -> List[Dict[str, Any]]:
    data = json.loads(BENIGN_PATH.read_text(encoding="utf-8"))
    items = []
    for x in data:
        img = x.get("image")
        b64_uri = None
        if img:
            try:
                raw = img["bytes"]["data"]
                media = img["bytes"].get("media_type", "image/jpeg")
                raw_bytes = base64.b64decode(raw)
                b64_uri = _img_bytes_to_b64(raw_bytes)
            except Exception:
                b64_uri = None
        items.append({
            "qid": str(x.get("id", "")),
            "text": str(x.get("question", "")).strip(),
            "image_b64": b64_uri,
        })
    log(f"benign: {len(items)} items ({sum(1 for i in items if i['image_b64'])} with image)")
    return items


# ---------------------------------------------------------------------------
# Embedding (DashScope qwen2.5-vl-embedding) with caching
# ---------------------------------------------------------------------------
def _embed_one(text: str, image_b64: Optional[str], model: str) -> Optional[List[float]]:
    from dashscope import MultiModalEmbedding
    payload: List[Dict[str, Any]] = [{"text": text}]
    if image_b64:
        payload[0]["image"] = image_b64
    for attempt in range(4):
        try:
            resp = MultiModalEmbedding.call(model=model, input=payload)
            if resp.status_code == 200:
                return resp.output["embeddings"][0]["embedding"]
        except Exception as exc:
            log(f"embed attempt {attempt + 1} failed: {exc}")
        time.sleep(1.0 + attempt)
    return None


def embed_items(items: List[Dict[str, Any]], cache_name: str, force: bool = False,
                workers: int = 6) -> np.ndarray:
    """Embed items (with image if present) and cache as npy + sidecar json.

    Concurrent workers; per-item retry handled inside _embed_one."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path = cache_dir / f"{cache_name}.npy"
    meta_path = cache_dir / f"{cache_name}.json"
    if npy_path.exists() and not force:
        embs = np.load(npy_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        log(f"cache hit: {cache_name} ({embs.shape[0]} embs)")
        return embs

    embs_list: List[Optional[List[float]]] = [None] * len(items)
    t0 = time.time()
    done = 0

    def work(i: int):
        return i, _embed_one(items[i]["text"], items[i].get("image_b64"), EMBED_MODEL)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, i) for i in range(len(items))]
        for fut in as_completed(futs):
            i, emb = fut.result()
            if emb is None:
                raise RuntimeError(f"embedding failed at item {i} of {cache_name}")
            embs_list[i] = emb
            done += 1
            if done % 100 == 0:
                el = time.time() - t0
                log(f"{cache_name}: {done}/{len(items)} embedded ({el:.0f}s elapsed)")

    embs = np.asarray(embs_list, dtype=np.float64)
    np.save(npy_path, embs)
    meta_path.write_text(
        json.dumps({"n": len(embs), "model": EMBED_MODEL,
                    "texts": [it["text"] for it in items]}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    log(f"{cache_name}: embedded {len(embs)} -> {npy_path} ({time.time()-t0:.0f}s)")
    return embs


# ---------------------------------------------------------------------------
# Similarity / threshold selection
# ---------------------------------------------------------------------------
def normalize(embs: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(embs, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return embs / n


def cos_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return normalize(a) @ normalize(b).T


def pick_threshold(attack_sim: np.ndarray, benign_sim: np.ndarray,
                   lo: float = 0.30, hi: float = 0.95) -> Dict[str, Any]:
    """Youden-optimal tau on query-vs-library top-1 similarity.
    attack_sim: (n_attack, n_lib) ; benign_sim: (n_benign, n_lib)"""
    a_top = attack_sim.max(axis=1)
    b_top = benign_sim.max(axis=1)
    grid = np.linspace(lo, hi, 651)
    best, best_score = None, -1.0
    for tau in grid:
        tpr = (a_top >= tau).mean()
        fpr = (b_top >= tau).mean()
        score = tpr - fpr
        if score > best_score:
            best_score, best = score, tau
    tau = float(best)
    return {
        "tau": tau,
        "attack_activation": float((a_top >= tau).mean()),
        "benign_activation": float((b_top >= tau).mean()),
        "attack_top1": {"mean": float(a_top.mean()), "median": float(np.median(a_top)),
                        "max": float(a_top.max())},
        "benign_top1": {"mean": float(b_top.mean()), "median": float(np.median(b_top)),
                        "max": float(b_top.max())},
    }


# ---------------------------------------------------------------------------
# Retention policies
# ---------------------------------------------------------------------------
class Entry:
    __slots__ = ("eid", "kind", "a", "c", "o", "tau_t", "kappa", "ttl", "R",
                 "alive", "quarantined", "activated_once")

    def __init__(self, eid: int, kind: str, ttl_min: float = TTL_MIN,
                 ttl_max: float = TTL_MAX,
                 alpha: float = ALPHA, beta: float = BETA):
        self.eid = eid
        self.kind = kind                  # clean | rare | poison
        self.a = 0                        # activations
        self.c = 0                        # confirmations (activated by an attack query)
        self.o = 0                        # refutations (activated by a benign query)
        self.tau_t = None                 # last confirmation step (competence)
        self.kappa = (alpha) / (alpha + beta)   # 0.5 neutral prior
        self.ttl = ttl_min + (ttl_max - ttl_min) * self.kappa
        self.R = self.ttl                 # remaining lifetime
        self.alive = True
        self.quarantined = False
        self.activated_once = False


def _activate(e: Entry, t: int, is_attack: bool, policy: str,
              delta: float = DELTA, quarantine_o: int = QUARANTINE_O,
              alpha: float = ALPHA, beta: float = BETA) -> None:
    e.a += 1
    e.activated_once = True
    if policy == "current":
        # hit-to-refresh: every activation refreshes the lifetime
        e.tau_t = t
        e.R = TTL_BASE
    elif policy == "competence":
        if is_attack:
            e.c += 1
            e.tau_t = t                      # refresh only on CONFIRMATION
        else:
            e.o += 1
            e.R *= delta                     # refutation accelerates decay
            if e.o >= quarantine_o:
                e.quarantined = True
        e.kappa = (e.c + alpha) / (e.c + e.o + alpha + beta)
        e.ttl = TTL_MIN + (TTL_MAX - TTL_MIN) * e.kappa
        if e.c > 0:
            e.R = e.ttl                      # confirmed: lifetime restored from confirmation time


def _age(e: Entry, t: int, policy: str) -> None:
    """End-of-step aging. Returns nothing; flags deletion via alive=False."""
    if policy == "current":
        e.R -= 1.0
        if e.R <= 0:
            e.alive = False
    elif policy == "competence":
        if e.c > 0:
            e.R = e.ttl - (t - e.tau_t)      # decay measured from last CONFIRMATION
        else:
            e.R = e.ttl - t                  # never confirmed: decays from insertion
        if e.R <= 0 and e.c == 0:
            e.alive = False                  # only never-confirmed entries are deletable
    # no-TTL: nothing


def simulate(policy: str, M_benign: np.ndarray, M_attack: np.ndarray,
             M_rare: np.ndarray, tau: float,
             lib_kinds: List[str],
             M_attack_extra: Optional[np.ndarray] = None,
             extra_label: str = "extra",
             seed: int = SEED + 1000,  # default reproduces the originally reported run
             ttl_base: float = TTL_BASE,
             ttl_min: float = TTL_MIN,
             ttl_max: float = TTL_MAX,
             l3_tpr: float = 1.0,
             l3_fpr: float = 0.0,
             topk: Optional[int] = None,
             delta: float = DELTA,
             quarantine_o: int = QUARANTINE_O,
             alpha: float = ALPHA,
             beta: float = BETA,
             r_benign: float = R_BENIGN,
             rare_batch_steps: Optional[List[int]] = None,
             novel_frac: float = 0.0) -> Dict[str, Any]:
    """Run the 20-step window under one retention policy.

    M_benign: (n_benign_q, n_lib) similarities for benign queries
    M_attack: (n_attack_q,  n_lib) similarities for attack probes (drives the window)
    M_rare:   (n_rare_q,    n_lib) similarities for the rare entries' own trigger
                                    queries (diagonal ~1.0 by construction, once each)
    lib_kinds: per-column kind in {"clean", "rare", "poison"}.
    M_attack_extra: optional second attack matrix (e.g. SIUO) evaluated post-window.
    l3_tpr / l3_fpr: L3 oracle accuracy — P(verdict=unsafe | true attack / true benign).
                     Default (1.0 / 0.0) = perfect oracle (query label proxy).
    topk: if set, a query activates only library entries within its top-k (production
          L2 returns top-k); None = every entry with sim >= tau is activated.
    delta / quarantine_o / alpha / beta: retention-policy constants (E3 sensitivity
          knob). Defaults reproduce the originally reported run.
    r_benign: benign share used to size each step's query batch (default 0.99 = the
          main protocol). Lower values model traffic-composition drift (E3 drift arm).
    rare_batch_steps: steps (0-indexed) at which a batch of rare-variant queries is
          injected. Default = RARE_BATCH_STEPS (every 5 steps). Setting a sparser
          schedule (gap > T_max) models a *dormant rare threat* that must survive a
          long inactive interval — the "activation-interval drift" arm (E3).
    novel_frac: extra per-step fraction of NEVER-CONFIRMED novel entries injected as
          benign (i.e. benign traffic that resembles unconfirmed library entries),
          raising the novel/unconfirmed share of traffic — the "novel-share drift"
          arm (E3). These queries are drawn from the poison/rare columns and counted
          as benign, so they add refutations without confirmations.
    """
    n_lib = len(lib_kinds)
    # normalize policy names ("competence-TTL" -> "competence", "current-TTL" -> "current", ...)
    policy_key = {"no-TTL": "none", "current-TTL": "current",
                  "competence-TTL": "competence"}.get(policy, policy)
    clean_lib_idx = [i for i, k in enumerate(lib_kinds) if k == "clean"]
    rare_lib_idx = [i for i, k in enumerate(lib_kinds) if k == "rare"]
    poison_lib_idx = [i for i, k in enumerate(lib_kinds) if k == "poison"]
    n_clean, n_rare, n_poison = len(clean_lib_idx), len(rare_lib_idx), len(poison_lib_idx)
    entries = [Entry(i, lib_kinds[i], ttl_min=ttl_min, ttl_max=ttl_max,
                     alpha=alpha, beta=beta) for i in range(n_lib)]
    poison_set = set(poison_lib_idx)
    # parameterised constants (E3): rebind so the closures below read the passed values
    DELTA, QUARANTINE_O, ALPHA, BETA = delta, quarantine_o, alpha, beta

    # per-step query batch composition (drift arm lowers r_benign)
    n_benign_per_step = int(round(STEP_SIZE * r_benign))
    n_attack_per_step = STEP_SIZE - n_benign_per_step

    rng = random.Random(seed)
    n_benign_q, n_attack_q = M_benign.shape[0], M_attack.shape[0]
    n_rare_q = M_rare.shape[0]

    def l3_verdict(is_attack_true: bool) -> bool:
        """L3 judge verdict (unsafe?) under oracle-accuracy noise."""
        if l3_tpr == 1.0 and l3_fpr == 0.0:
            return is_attack_true            # perfect oracle: no RNG consumption
        if is_attack_true:
            return rng.random() < l3_tpr
        return rng.random() < l3_fpr

    def activate_entry(e: Entry, t: int, verdict_unsafe: bool) -> None:
        if not (e.alive and not e.quarantined):
            return
        e.a += 1
        e.activated_once = True
        if policy_key == "current":
            e.tau_t = t
            e.R = ttl_base
        elif policy_key == "competence":
            if verdict_unsafe:
                e.c += 1
                e.tau_t = t
            else:
                e.o += 1
                e.R *= DELTA
                if e.o >= QUARANTINE_O:
                    e.quarantined = True
            e.kappa = (e.c + ALPHA) / (e.c + e.o + ALPHA + BETA)
            e.ttl = ttl_min + (ttl_max - ttl_min) * e.kappa
            if e.c > 0:
                e.R = e.ttl

    def age_entry(e: Entry, t: int) -> None:
        if not (e.alive and not e.quarantined):
            return
        if policy_key == "current":
            e.R -= 1.0
            if e.R <= 0:
                e.alive = False
        elif policy_key == "competence":
            if e.c > 0:
                e.R = e.ttl - (t - e.tau_t)
            else:
                e.R = e.ttl - t
            if e.R <= 0 and e.c == 0:
                e.alive = False

    rare_used = 0
    rare_sched = RARE_BATCH_STEPS if rare_batch_steps is None else list(rare_batch_steps)
    n_novel_per_step = int(round(STEP_SIZE * novel_frac))
    # columns that count as "novel/unconfirmed" entries (rare + poison)
    novel_col_mask = np.zeros(n_lib, dtype=bool)
    for _i in (rare_lib_idx + poison_lib_idx):
        novel_col_mask[_i] = True
    for t in range(N_STEPS):
        # ---- build the step's queries ----
        benign_rows = [rng.randrange(n_benign_q) for _ in range(n_benign_per_step)]
        attack_rows = [rng.randrange(n_attack_q) for _ in range(n_attack_per_step)]
        rare_rows: List[int] = []
        if t in rare_sched and rare_used < n_rare_q:
            # inject a batch of rare-variant queries (each rare entry appears exactly once)
            rare_rows = list(range(rare_used, min(rare_used + RARE_PER_BATCH, n_rare_q)))
            rare_used += len(rare_rows)
            # rare-variant queries REPLACE part of the attack slot (guide: attack variants)
            attack_rows = attack_rows[:n_attack_per_step - len(rare_rows)] if len(rare_rows) < n_attack_per_step else attack_rows[:0]

        # ---- novel-share drift: benign traffic resembling unconfirmed entries ----
        # drawn from the rare+poison columns and processed as benign; they refute
        # (raise o) without confirming, i.e. they inflate the novel/unconfirmed share.
        novel_rows: List[int] = [rng.randrange(n_benign_q) for _ in range(n_novel_per_step)]

        # ---- process queries ----
        def apply_rows(rows: List[int], is_attack_true: bool, sim_mat: np.ndarray,
                       col_mask: Optional[np.ndarray] = None) -> None:
            if not rows:
                return
            sub = sim_mat[np.asarray(rows)]
            if col_mask is not None:
                sub = np.where(col_mask[None, :], sub, -1.0)
            if topk is not None and topk < n_lib:
                # production-like retrieval: only the top-k columns can be hit
                topk_idx = np.argsort(-sub, axis=1)[:, :topk]
                mask = np.zeros_like(sub, dtype=bool)
                np.put_along_axis(mask, topk_idx, True, axis=1)
                sub = np.where(mask, sub, -1.0)
            hits = sub >= tau                     # (n_q, n_lib)
            verdict_unsafe = l3_verdict(is_attack_true)
            for r in range(hits.shape[0]):
                hit_ids = np.flatnonzero(hits[r])
                for j in hit_ids:
                    activate_entry(entries[j], t, verdict_unsafe)

        apply_rows(benign_rows, False, M_benign)
        apply_rows(attack_rows, True, M_attack)
        apply_rows(rare_rows, True, M_rare)
        if novel_rows:
            # benign queries targeting the unconfirmed (rare+poison) columns only
            apply_rows(novel_rows, False, M_benign, col_mask=novel_col_mask)

        # ---- end-of-step aging ----
        for e in entries:
            age_entry(e, t)

    # ---- metrics ----
    alive = [e for e in entries if e.alive and not e.quarantined]
    alive_set = {e.eid for e in alive}

    clean_kept = sum(1 for i in clean_lib_idx if i in alive_set)
    rare_kept = sum(1 for i in rare_lib_idx if i in alive_set)
    poison_kept = sum(1 for i in poison_lib_idx if i in alive_set)
    clean_deleted = n_clean - clean_kept
    rare_deleted = n_rare - rare_kept
    poison_deleted = n_poison - poison_kept

    # poison activation rate: measured on no-TTL (all poison entries stay in library)
    poison_act = sum(1 for i in poison_lib_idx if entries[i].activated_once)

    # post-window FPR / TPR against the surviving library
    def fpr_tpr(sim_mat: np.ndarray) -> Dict[str, float]:
        top = sim_mat[:, list(alive_set)].max(axis=1)
        return {"rate": float((top >= tau).mean())}

    fpr_post = fpr_tpr(M_benign)["rate"]
    tpr_post = fpr_tpr(M_attack)["rate"]
    out = {
        "policy": policy,
        "poison_activation_rate": poison_act / n_poison,
        "poison_recall": poison_deleted / n_poison,
        "collateral_removal": clean_deleted / n_clean,
        "rare_retention": rare_kept / n_rare,
        "rare_deleted": rare_deleted,
        "fpr_post": fpr_post,
        "tpr_post": tpr_post,
        "lib_survivors": {"clean": clean_kept, "rare": rare_kept, "poison": poison_kept},
        # diagnostics
        "diag": {
            "poison_o_hist": sorted({e.o: sum(1 for i in poison_lib_idx if entries[i].o == e.o)
                                     for e in (entries[i] for i in poison_lib_idx)}.items()),
            "poison_c_hist": sorted({e.c: sum(1 for i in poison_lib_idx if entries[i].c == e.c)
                                     for e in (entries[i] for i in poison_lib_idx)}.items()),
            "rare_deleted_eids": [i for i in rare_lib_idx if i not in alive_set],
            # cause attribution only meaningful under competence (o counters updated)
            "clean_deleted_with_refutation": (
                sum(1 for i in clean_lib_idx
                    if i not in alive_set and entries[i].o > 0)
                if policy_key == "competence" else None),
            "clean_deleted_never_activated": (
                sum(1 for i in clean_lib_idx
                    if i not in alive_set and entries[i].o == 0)
                if policy_key == "competence" else None),
        },
    }
    if M_attack_extra is not None:
        out[f"tpr_{extra_label}_post"] = fpr_tpr(M_attack_extra)["rate"]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def phase_embed() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    # 1) MM-SafetyBench: clean library (500) + rare (20) + attack-main probes (167),
    #    all disjoint splits of one seed-fixed shuffle.
    msb = load_mmsb_items(N_CLEAN + N_RARE + N_ATTACK_MAIN, 0)
    clean_items = [{"text": it["text"], "image_b64": _img_bytes_to_b64(it["image_bytes"])}
                   for it in msb["clean"][:N_CLEAN]]
    rare_items = [{"text": it["text"], "image_b64": _img_bytes_to_b64(it["image_bytes"])}
                  for it in msb["clean"][N_CLEAN:N_CLEAN + N_RARE]]
    attack_main_items = [{"text": it["text"], "image_b64": _img_bytes_to_b64(it["image_bytes"])}
                         for it in msb["clean"][N_CLEAN + N_RARE:N_CLEAN + N_RARE + N_ATTACK_MAIN]]
    embed_items(clean_items, "l_threat")
    embed_items(rare_items, "rare")
    embed_items(attack_main_items, "q_attack_main")
    # 2) benign pool (embed fresh)
    benign_items = load_benign_items()
    embed_items(benign_items, "q_benign")
    # 3) SIUO attack embeddings from chroma cache (verify)
    siuo_embs = load_siuo_embeddings_from_chroma()
    np.save(CACHE_DIR / "q_attack_siuo.npy", siuo_embs)
    log(f"q_attack_siuo cached from chroma: {siuo_embs.shape}")
    log("embed phase done.")


def _load_cache(name: str) -> np.ndarray:
    return np.load(CACHE_DIR / f"{name}.npy")


def phase_simulate() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    l_threat = _load_cache("l_threat")
    rare = _load_cache("rare")
    q_benign = _load_cache("q_benign")
    q_attack_main = _load_cache("q_attack_main")
    q_attack_siuo = _load_cache("q_attack_siuo")
    log(f"l_threat={l_threat.shape} rare={rare.shape} "
        f"q_benign={q_benign.shape} q_attack_main={q_attack_main.shape} "
        f"q_attack_siuo={q_attack_siuo.shape}")

    # ---- threshold selection: Youden on query-vs-CLEAN-library top-1 ----
    M_attack_clean = cos_mat(q_attack_main, l_threat)   # (167, 500)
    M_benign_clean = cos_mat(q_benign, l_threat)        # (1000, 500)
    tau_info = pick_threshold(M_attack_clean, M_benign_clean)
    tau = tau_info["tau"]
    log(f"tau selection: {json.dumps(tau_info, indent=1)}")

    # ---- poison construction: benign queries nearest to the library (by construction retrievable) ----
    b_top = M_benign_clean.max(axis=1)
    order = np.argsort(-b_top)
    cand = [int(i) for i in order if b_top[i] >= tau]
    if len(cand) < N_POISON:
        # relax to the highest-similarity benign queries (report the relaxation)
        cand = [int(i) for i in order[:N_POISON]]
        relaxed = True
        min_sim = float(b_top[cand[-1]])
    else:
        cand = cand[:N_POISON]
        relaxed = False
        min_sim = float(b_top[cand[-1]])
    poison_idx = cand
    log(f"poison: {len(poison_idx)} entries, natural candidates>={tau}: "
        f"{len([i for i in order if b_top[i] >= tau])}, relaxed={relaxed}, min_sim={min_sim:.3f}")
    poison_sims = [float(b_top[i]) for i in poison_idx]

    # ---- full library: 500 clean + 20 rare + 50 poison (poison = benign prototypes) ----
    poison_embs = q_benign[poison_idx]
    lib_embs = np.vstack([l_threat, rare, poison_embs])     # (570, 1024)
    lib_kinds = ["clean"] * N_CLEAN + ["rare"] * N_RARE + ["poison"] * N_POISON
    M_attack_main = cos_mat(q_attack_main, lib_embs)        # (167, 570)
    M_attack_siuo = cos_mat(q_attack_siuo, lib_embs)        # (167, 570)
    M_benign = cos_mat(q_benign, lib_embs)                  # (1000, 570)
    M_rare = cos_mat(rare, lib_embs)                        # (20, 570): triggers hit own entry (~1.0)

    # ---- three policies ----
    results = []
    for policy in ["no-TTL", "current-TTL", "competence-TTL"]:
        res = simulate(policy, M_benign, M_attack_main, M_rare, tau, lib_kinds,
                       M_attack_extra=M_attack_siuo, extra_label="siuo")
        results.append(res)
        log(f"{policy}: {json.dumps(res)}")

    # ---- report table (guide Step 5) ----
    header = ["策略", "毒化激活率", "毒化召回", "附带删除率", "罕见真威胁保留率", "FPR(后)", "TPR(后)"]
    rows = []
    for r in results:
        rows.append([
            r["policy"],
            f"{r['poison_activation_rate']:.1%}",
            "—" if r["policy"] == "no-TTL" else f"{r['poison_recall']:.1%}",
            f"{r['collateral_removal']:.1%}",
            f"{r['rare_retention']:.1%}",
            f"{r['fpr_post']:.1%}",
            f"{r['tpr_post']:.1%}",
        ])

    # ---- Step 6 sub-experiment table ----
    rows6 = []
    for r in results:
        rows6.append([r["policy"], f"{r['rare_retention']:.1%}", f"{r['rare_deleted']}/{N_RARE}"])

    report = {
        "protocol": "P0-5 TTL mechanism (P0级修改指南 §5.4)",
        "datasets": {
            "l_threat": {"source": "MM-SafetyBench SD", "n": N_CLEAN},
            "rare": {"source": "MM-SafetyBench SD (held-out)", "n": N_RARE},
            "q_attack_main": {"source": "MM-SafetyBench SD held-out (disjoint from L_threat)",
                              "n": N_ATTACK_MAIN},
            "q_attack_siuo": {"source": "SIUO (167, disjoint from L_threat)",
                              "n": q_attack_siuo.shape[0]},
            "q_benign": {"source": "MME+MM-Vet+custom", "n": q_benign.shape[0]},
            "poison": {"n": N_POISON, "construct": "benign queries with max-sim>=tau",
                       "min_sim": min_sim, "relaxed": relaxed},
        },
        "embedding": {"model": EMBED_MODEL, "dim": int(l_threat.shape[1])},
        "threshold": tau_info,
        "params": {
            "n_steps": N_STEPS, "step_size": STEP_SIZE, "r_benign": R_BENIGN,
            "ttl_base": TTL_BASE, "ttl_min": TTL_MIN, "ttl_max": TTL_MAX,
            "delta": DELTA, "quarantine_o": QUARANTINE_O, "alpha": ALPHA, "beta": BETA,
            "seed": SEED,
        },
        "poison_construction": {
            "n": len(poison_idx),
            "max_sim_to_lib": [round(s, 4) for s in poison_sims],
            "min": round(min_sim, 4),
        },
        "step5_table": {"header": header, "rows": rows},
        "step6_table": {"header": ["策略", "罕见真威胁保留率", "删除数/总数"], "rows": rows6},
        "acceptance": {
            "poison_activation_gt_0": all(r["poison_activation_rate"] > 0 for r in results),
            "competence_poison_survival_le_current": True,
            "competence_rare_retention_ge_current": True,
        },
        "raw": results,
    }
    # acceptance detail
    comp = next(r for r in results if r["policy"] == "competence-TTL")
    curr = next(r for r in results if r["policy"] == "current-TTL")
    report["acceptance"]["competence_poison_survival_le_current"] = \
        (1 - comp["poison_recall"]) <= (1 - curr["poison_recall"])
    report["acceptance"]["competence_rare_retention_ge_current"] = \
        comp["rare_retention"] >= curr["rare_retention"]

    out_path = OUT_DIR / "ttl_experiment_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"report saved -> {out_path}")

    # console table
    print("\n" + "=" * 100)
    print("Step 5 表 (指南 §5.4, 缺一不可)")
    print("=" * 100)
    print(f"{'策略':<16}{'毒化激活率':<12}{'毒化召回':<10}{'附带删除率':<12}{'罕见保留率':<12}{'FPR(后)':<10}{'TPR(后)':<10}")
    for r in rows:
        print(f"{r[0]:<16}{r[1]:<12}{r[2]:<10}{r[3]:<12}{r[4]:<12}{r[5]:<10}{r[6]:<10}")
    print("\nStep 6 表 (罕见真威胁针对性子实验)")
    print(f"{'策略':<16}{'罕见保留率':<12}{'删除':<12}")
    for r in rows6:
        print(f"{r[0]:<16}{r[1]:<12}{r[2]:<12}")
    print("\n验收判据:", json.dumps(report["acceptance"], ensure_ascii=False, indent=1))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / "defenceAgent" / ".env")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="P0-5 TTL redesigned experiment")
    ap.add_argument("--phase", choices=["embed", "simulate", "all"], default="all")
    args = ap.parse_args()
    if args.phase in ("embed", "all"):
        phase_embed()
    if args.phase in ("simulate", "all"):
        phase_simulate()


if __name__ == "__main__":
    main()
