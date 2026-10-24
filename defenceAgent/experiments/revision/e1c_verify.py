# -*- coding: utf-8 -*-
"""Verify EX-E1-③ aggregated report: recompute ASR/CI from the checkpoint and
assert they match benchmark_full_report.json. Read-only; no API calls.

Also prints the HarmBench copyright-memorisation exclusion view, so the
safety-relevant ASR can be quoted separately.
"""
from __future__ import annotations
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "results" / "e1c_ckpt.json"
REP = ROOT / "results" / "benchmark_full_report.json"


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ck = json.loads(CKPT.read_text(encoding="utf-8"))["results"]
    rep = json.loads(REP.read_text(encoding="utf-8"))
    rows = list(ck.values())
    print(f"checkpoint samples: {len(rows)}")
    print("verdicts:", Counter(r.get("verdict") for r in rows))

    ok = True
    for b in sorted({r["benchmark"] for r in rows}):
        sub = [r for r in rows if r["benchmark"] == b]
        n = len(sub)
        k = sum(1 for r in sub if r["action"] == "allow")
        p = k / n if n else 0
        lo, hi = wilson(k, n)
        rr = rep["results"].get(b)
        match = (rr and rr["N"] == n and rr["allow"] == k
                 and abs(rr["asr"]["p"] - p) < 1e-9
                 and abs(rr["asr"]["wilson95"][0] - lo) < 1e-9
                 and abs(rr["asr"]["wilson95"][1] - hi) < 1e-9)
        ok &= bool(match)
        print(f"  {b}: recomputed N={n} allow={k} ASR={p*100:.2f}% "
              f"CI=[{lo*100:.2f},{hi*100:.2f}]  report_match={match}")

    # copyright-memorisation exclusion view
    hb = [r for r in rows if r["benchmark"] == "HarmBench"]
    hb_nc = [r for r in hb if r["config"] != "copyright"]
    for label, s in [("HarmBench all", hb), ("HarmBench excl. copyright", hb_nc)]:
        n = len(s)
        k = sum(1 for r in s if r["action"] == "allow")
        lo, hi = wilson(k, n)
        print(f"  {label}: N={n} allow={k} ASR={k/n*100 if n else 0:.2f}% CI=[{lo*100:.2f},{hi*100:.2f}]")

    print("VERIFY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
