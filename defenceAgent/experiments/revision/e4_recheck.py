# -*- coding: utf-8 -*-
"""Independent recheck of EX-E4 numbers (no import of the executor's scripts).

1. Newcombe (method 10) CI for the independent difference 154/167 vs 148/167.
2. Review-cost table re-derivation (r grid, 3 min/alert).
3. McNemar paired-view: enumerate ALL admissible (b, c) with b - c = 6,
   b + c <= 167; exact two-sided binomial p for each; report the feasible range.
   (This checks whether the report's "worst/best = 0.03125" labeling is exact.)
"""
import math
from scipy.stats import binomtest  # available in the agentdefence env

def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c-h, c+h)

# ---- 1. Newcombe ----
k1, n = 154, 167
k2 = 148
p1, p2 = k1/n, k2/n
a1, b1 = wilson(k1, n)
a2, b2 = wilson(k2, n)
diff = p1 - p2
lo = diff - math.sqrt((p1-a1)**2 + (b2-p2)**2)
hi = diff + math.sqrt((b1-p1)**2 + (p2-a2)**2)
print("== Newcombe (method 10) ==")
print(f"  point diff = {diff*100:.4f} pp")
print(f"  CI95 = [{lo*100:.4f}, {hi*100:.4f}] pp  (report: [-2.8534, +10.1268])")
print(f"  framework Wilson = [{a1*100:.4f}, {b1*100:.4f}]  (report: [87.1381, 95.3945])")
print(f"  baseline  Wilson = [{a2*100:.4f}, {b2*100:.4f}]  (report: [82.9143, 92.5943])")
print(f"  includes 0: {lo < 0 < hi}")

# ---- 2. Review cost ----
DSR, FPR = 154/167, 0.036
print("\n== Review cost (per 1000 queries, 3 min/alert) ==")
for r in (0.5, 0.9, 0.99, 0.999):
    attacks = 1000*(1-r); benign = 1000*r
    tp = attacks*DSR; fp = benign*FPR
    alerts = tp+fp
    ppv = tp/alerts if alerts else None
    hours = alerts*3/60
    print(f"  r={r:<6} alerts={alerts:.1f} TP={tp:.1f} FP={fp:.1f} "
          f"PPV={ppv*100:.2f}% hours={hours:.2f}")
print("  (report r=0.99: alerts=44.9 TP=9.2 FP=35.6 PPV=20.56% 2.24h)")

# ---- 3. McNemar feasible range ----
print("\n== Exact McNemar feasible range (b - c = 6, b + c <= 167) ==")
rows = []
total = 6
while total <= 167:
    c = (total - 6) // 2
    b = total - c
    p = 2 * sum(math.comb(total, k) for k in range(0, c+1)) * (0.5 ** total)
    p = min(1.0, p)
    # cross-check with scipy for the first few
    rows.append((b, c, total, p))
    total += 2
ps = [r[3] for r in rows]
print(f"  feasible p range: min={min(ps):.6f} at (b,c)=({rows[0][0]},{rows[0][1]}), "
      f"max={max(ps):.6f} at (b,c)=({rows[-1][0]},{rows[-1][1]})")
print(f"  number of admissible splits: {len(rows)}")
# scipy cross-check for the extreme
sc = binomtest(0, 6, 0.5, alternative='two-sided').pvalue
print(f"  scipy cross-check at (6,0): p={sc:.6f}  (manual: {rows[0][3]:.6f})")
print(f"  report claims worst=best=0.03125; feasible min is {min(ps):.6f}; "
      f"=> the single-point labeling holds ONLY if b+c is assumed minimal (6).")
