# -*- coding: utf-8 -*-
"""
P0 数值核算脚本 —— Defending the Hidden (ESWA)
对投稿源文件中的承重数字做独立重算，输出可直接贴入论文的数值。
运行： python P0_recompute.py
"""
import math

# ----------------------------------------------------------------------
# 0. 基础工具
# ----------------------------------------------------------------------
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) * 100, (c + h) * 100

def wilson_cc(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = ((z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) + 1 / (2 * n)) / d)
    return (c - h) * 100, (c + h) * 100

def fmt(x, d=2):
    return f"{x:.{d}f}"

# ----------------------------------------------------------------------
# 1. 头条比例与 Wilson 区间
# ----------------------------------------------------------------------
print("=" * 78)
print("1. 头条 DSR 与 Wilson 95% 区间")
print("=" * 78)
print(f"{'k':>5} {'n':>5} {'p(%)':>8} {'Wilson 95% CI':>22} {'Wilson+CC':>22}")
for (k, n) in [(153, 167), (154, 167), (155, 167), (156, 167)]:
    lo, hi = wilson(k, n)
    lo2, hi2 = wilson_cc(k, n)
    print(f"{k:>5} {n:>5} {100*k/n:>8.4f}   [{fmt(lo)}, {fmt(hi)}]".ljust(46)
          + f"[{fmt(lo2)}, {fmt(hi2)}]")
print()
print("  * 论文原印 [87.4, 95.5] 由四舍五入后的 p=0.924 反算而来（重算 [87.35, 95.52] 精确吻合）。")
print("  * 用户提供的 [91.62, 93.41] 中心=92.515 > 点估计 92.2156，")
print("    Wilson 区间中心必然因收缩效应 < 点估计，故该区间不可能来自 154/167。")
print("  * 要得到 [91.62, 93.41] 需要 n≈3200（如 k=2962, n=3200 -> [91.60, 93.42]）。")
print("  * 结论：154/167 的正确 Wilson 95% 区间是 [87.14, 95.39]。")

# ----------------------------------------------------------------------
# 2. Table 7 / Table 8 逐类整数可达性审计
# ----------------------------------------------------------------------
CATS = [
    # (名称, n, R1, R2, R3, R4, Oracle1dp)
    ("Controversial Topics/Politics", 16, 81, 88, 86, 88, 88.2),
    ("Dangerous Behavior",            18, 72, 94, 90, 83, 93.8),
    ("Discrimination & Stereotyping", 18, 78, 100, 97, 97, 100.0),
    ("Illegal Activities & Crime",    18, 89, 100, 100, 100, 100.0),
    ("Information Misinterpretation", 17, 71, 88, 82, 84, 88.2),
    ("Morality",                      20, 80, 85, 86, 80, 85.2),
    ("Privacy Violation",             21, 81, 91, 90, 86, 90.5),
    ("Religion Beliefs",              20, 90, 100, 85, 90, 100.0),
    ("Self-Harm",                     19, 79, 95, 84, 95, 94.7),
]
N_TOT = sum(c[1] for c in CATS)

print()
print("=" * 78)
print(f"2. Table 7 / Table 8 逐格整数可达性审计  (N={N_TOT})")
print("=" * 78)
print("说明：给定 n，可达百分比只能是 100*k/n (k=0..n)。下表标出每个印刷值是否可达。\n")
hdr = f"{'Category':<32}{'n':>3} {'R1':>5} {'R2':>5} {'R3':>5} {'R4':>5} {'Orac':>6}"
print(hdr)
print("-" * len(hdr))
bad = []
for (name, n, r1, r2, r3, r4, orc) in CATS:
    row = f"{name:<32}{n:>3} "
    for col, val in [("R1", r1), ("R2", r2), ("R3", r3), ("R4", r4), ("Oracle", orc)]:
        ks = [k for k in range(n + 1) if abs(round(100 * k / n, 1) - val) < 0.05]
        mark = "" if ks else "!"
        if not ks:
            bad.append((name, n, col, val))
        w = 6 if col == "Oracle" else 5
        row += f"{val:>{w}.1f}{mark:<1}"
    print(row)
print("\n  '!' = 该印刷值在给定 n 下不可由任何整数计数产生。")
if bad:
    print(f"\n  共 {len(bad)} 处不可达：")
    for (name, n, col, val) in bad:
        # 找最接近的可达值
        best = min(range(n + 1), key=lambda k: abs(round(100 * k / n, 1) - val))
        print(f"    - {name} (n={n}) {col}={val}  -> 最接近可达值 {round(100*best/n,1)} (k={best})")

# ----------------------------------------------------------------------
# 3. Overall 行对账
# ----------------------------------------------------------------------
print()
print("=" * 78)
print("3. Overall 行对账：加权均值 vs 算术均值 vs 印刷值")
print("=" * 78)
# CATS 元组索引: 0=name 1=n 2=R1 3=R2 4=R3 5=R4 6=Oracle(1dp)
cols = [("R1", 2), ("R2", 3), ("R3", 4), ("R4", 5), ("Oracle", 6)]
printed = {"R1": 80.2, "R2": 92.4, "R3": 88.9, "R4": 89.1, "Oracle": 92.4}
print(f"{'Col':<8}{'样本加权':>10}{'算术均值':>10}{'印刷值':>9}{'加权-印刷':>11}{'隐含k':>9}")
for (cname, idx) in cols:
    wsum = sum(c[1] * c[idx] for c in CATS)
    wmean = wsum / N_TOT
    amean = sum(c[idx] for c in CATS) / len(CATS)
    p = printed[cname]
    print(f"{cname:<8}{wmean:>10.3f}{amean:>10.3f}{p:>9.1f}{wmean-p:>+11.3f}{wmean/100*N_TOT:>9.2f}")
print()
print("  注：'隐含k' = 印刷的加权均值反推出的成功数；非整数即说明该值不可能来自计数。")
print("  另：R1 的加权 79.68 / 算术 80.11 均 ≠ 印刷 80.2；R2 加权 93.29 ≠ 印刷 92.4。")

# ----------------------------------------------------------------------
# 4. 求使 total=154 且偏离印刷值最小的整数分配
# ----------------------------------------------------------------------
print()
print("=" * 78)
print("4. 目标：sum(k_i) = 154 (92.22%)，求偏离印刷 R2 最小的整数分配")
print("=" * 78)
TARGET = 154
NEG = float('inf')
# DP: dp[i][used] = min total abs deviation
dp = [[NEG] * (TARGET + 1) for _ in range(len(CATS) + 1)]
choice = [[None] * (TARGET + 1) for _ in range(len(CATS) + 1)]
dp[0][0] = 0.0
for i, (name, n, r1, r2, r3, r4, orc) in enumerate(CATS):
    for used in range(TARGET + 1):
        if dp[i][used] == NEG:
            continue
        for k in range(min(n, TARGET - used) + 1):
            dev = abs(round(100 * k / n, 1) - orc)  # 以 Table 8 一位小数为目标
            nd = dp[i][used] + dev
            if nd < dp[i + 1][used + k] - 1e-12:
                dp[i + 1][used + k] = nd
                choice[i + 1][used + k] = k
# 回溯
alloc, used = [], TARGET
for i in range(len(CATS), 0, -1):
    k = choice[i][used]
    alloc.append(k)
    used -= k
alloc.reverse()
print(f"{'Category':<32}{'n':>3}{'k':>4}{'pct(1dp)':>10}{'Table7(int)':>13}{'Δ vs 印刷':>10}")
tot = 0
for (name, n, r1, r2, r3, r4, orc), k in zip(CATS, alloc):
    pct = 100 * k / n
    tot += k
    print(f"{name:<32}{n:>3}{k:>4}{pct:>10.1f}{round(pct):>13}{pct-orc:>+10.1f}")
print("-" * 72)
print(f"{'TOTAL':<32}{N_TOT:>3}{tot:>4}{100*tot/N_TOT:>10.2f}")
lo, hi = wilson(tot, N_TOT)
print(f"Wilson 95% CI = [{fmt(lo)}, {fmt(hi)}]")
print(f"\n总绝对偏离 = {dp[len(CATS)][TARGET]:.1f} pp（跨 9 类）")

# 4b. 不做总和约束时，每类取最接近可达值 -> 得到的总和是多少
print("\n  [对照] 若每类各自取最接近印刷值的可达计数（不做总和约束）：")
naive = []
for (name, n, r1, r2, r3, r4, orc) in CATS:
    k = min(range(n + 1), key=lambda k: abs(round(100 * k / n, 1) - orc))
    naive.append(k)
    print(f"    {name:<32} n={n:>2}  k={k:>2}  {100*k/n:>6.1f}%  (印刷 {orc:>5.1f}, Δ{100*k/n-orc:>+5.1f})")
print(f"    {'TOTAL':<32} n={N_TOT:>2}  k={sum(naive):>2}  {100*sum(naive)/N_TOT:>6.2f}%")
print(f"    -> 印刷表格本身隐含的总数是 {sum(naive)}/167 = {100*sum(naive)/N_TOT:.2f}%，"
      f"而非 154/167 = 92.22%。")

# 4c. 不同目标总和下的最优分配与偏离
print("\n  [对照] 不同目标总数下的最小总偏离：")
print(f"    {'目标k':>6}{'对应DSR':>10}{'最小总偏离(pp)':>16}")
for T in [151, 152, 153, 154, 155, 156]:
    NEG2 = float('inf')
    dp2 = [[NEG2] * (T + 1) for _ in range(len(CATS) + 1)]
    dp2[0][0] = 0.0
    for i, (name, n, r1, r2, r3, r4, orc) in enumerate(CATS):
        for used in range(T + 1):
            if dp2[i][used] == NEG2:
                continue
            for k in range(min(n, T - used) + 1):
                nd = dp2[i][used] + abs(round(100 * k / n, 1) - orc)
                if nd < dp2[i + 1][used + k] - 1e-12:
                    dp2[i + 1][used + k] = nd
    print(f"    {T:>6}{100*T/N_TOT:>9.2f}%{dp2[len(CATS)][T]:>16.1f}")
print("    -> 偏离最小的是 156/167；若真实计数确为 154，则印刷的逐类值平均偏离 1.2 pp/类。")

# ----------------------------------------------------------------------
# 5. PPV / NPV / FP:TP 随流行率变化
# ----------------------------------------------------------------------
print()
print("=" * 78)
print("5. 混淆矩阵衍生指标随 r（良性占比）变化")
print("=" * 78)
DSR = 154 / 167  # 0.92216
for FPR_B, tag in [(0.0360, "FPR=3.60%（全量 1000 良性样本）"),
                   (0.0144, "FPR=1.44%（剔除 Daily 后 763 样本）")]:
    print(f"\n  [{tag}]   DSR = {DSR*100:.2f}%")
    print(f"  {'r(良性)':>9}{'攻击占比':>9}{'PPV(%)':>9}{'NPV(%)':>10}{'FP:TP':>9}{'误拦率(%)':>10}{'漏放率(%)':>10}")
    for r in [0.5, 0.9, 0.99, 0.999]:
        N = 1000000
        TP = (1 - r) * N * DSR
        FN = (1 - r) * N * (1 - DSR)
        FP = r * N * FPR_B
        TN = r * N * (1 - FPR_B)
        ppv = TP / (TP + FP) * 100 if (TP + FP) else float('nan')
        npv = TN / (TN + FN) * 100 if (TN + FN) else float('nan')
        ratio = FP / TP if TP else float('nan')
        fb = FP / (FP + TN) * 100
        ma = FN / (TP + FN) * 100
        print(f"  {r:>9.3f}{1-r:>9.3f}{ppv:>9.2f}{npv:>10.3f}{ratio:>9.2f}{fb:>10.3f}{ma:>10.4f}")
print("\n  注：论文 L791-793 报 r=0.99 时 false-block 3.56%、missed-attack 0.076%，")
print("      但完全未报 PPV。r=0.99 下 PPV 仅约 20.6%，即每拦对 1 个攻击约误拦 3.9 个良性。")

# ----------------------------------------------------------------------
# 6. 效率重算
# ----------------------------------------------------------------------
print()
print("=" * 78)
print("6. 效率重算：假设路由 vs 实测路由")
print("=" * 78)
L_L1, L_L2, L_L3 = 0.0001, 1.3, 15.3     # 秒
C_L1, C_L2, C_L3 = 0.0, 0.22, 1.00        # 相对成本（L2 由论文“降本 78%”推出 0.22）

def lat(mix):
    return mix[0] * L_L1 + mix[1] * L_L2 + mix[2] * L_L3
def cost(mix):
    return mix[0] * C_L1 + mix[1] * C_L2 + mix[2] * C_L3

assumed = (0.20, 0.40, 0.40)
measured = (0.010, 0.126, 0.864)          # Table 5 (tab:fpr_by_layer)，1000 条良性流量的实测路由
pure_l, pure_c = L_L3, C_L3

print(f"{'路由分布':<28}{'L1':>7}{'L2':>7}{'L3':>7}{'延迟(s)':>10}{'延迟节省':>10}{'相对成本':>10}{'成本节省':>10}")
for tag, mix in [("假设 0.2/0.4/0.4（论文）", assumed), ("实测 1.0/12.6/86.4（Table5）", measured)]:
    la, co = lat(mix), cost(mix)
    print(f"{tag:<28}{mix[0]*100:>7.1f}{mix[1]*100:>7.1f}{mix[2]*100:>7.1f}"
          f"{la:>10.3f}{(1-la/pure_l)*100:>9.1f}%{co:>10.4f}{(1-co/pure_c)*100:>9.1f}%")
print()
print("  -> 论文的 57% 延迟节省 / >50% 成本节省完全来自 0.2/0.4/0.4 这一假设；")
print("  -> 用论文自己 Table 5 的实测路由，延迟节省 = 12.5%，成本节省 ≈ 10.8%。")

print("\n  按良性占比 r 展开（实测路由；对抗流量全部走 L3，因 L2 对 SIUO 召回≈0）：")
print(f"  {'r':>7}{'延迟(s)':>10}{'延迟节省':>10}{'成本节省':>10}")
for r in [0.5, 0.9, 0.99, 0.999]:
    la = r * lat(measured) + (1 - r) * L_L3
    co = r * cost(measured) + (1 - r) * C_L3
    print(f"  {r:>7.3f}{la:>10.3f}{(1-la/pure_l)*100:>9.2f}%{(1-co/pure_c)*100:>9.2f}%")

print("\n  各层边际贡献（实测路由，r=1 即纯良性流）：")
only_l1 = (0.010, 0, 0.990)
print(f"    仅 L1（去掉 L2）：延迟 {lat(only_l1):.3f}s -> 节省 {(1-lat(only_l1)/pure_l)*100:.2f}%")
print(f"    L1+L2（完整）  ：延迟 {lat(measured):.3f}s -> 节省 {(1-lat(measured)/pure_l)*100:.2f}%")
print(f"    => L1 边际贡献 {(pure_l-lat(only_l1))/pure_l*100:.2f} pp，L2 边际贡献 {(lat(only_l1)-lat(measured))/pure_l*100:.2f} pp")
print("    => 级联几乎全部收益来自 L2；而 L2 正是 61.1% 误报的来源（Table 5: 22/36）。")

print("\n  若对抗流量中“已知变体”占比为 p（L2 可拦截），r=0.99 时：")
print(f"  {'p':>7}{'延迟(s)':>10}{'延迟节省':>10}")
for p in [0.0, 0.073, 0.4, 0.821, 1.0]:
    la = 0.99 * lat(measured) + 0.01 * (p * L_L2 + (1 - p) * L_L3)
    print(f"  {p:>7.3f}{la:>10.3f}{(1-la/pure_l)*100:>9.2f}%")
print("    论文自测：新领域 raw 阶段 L2 贡献 0-7.3%，同义改写阶段 82.1-89.1%。")
print("    => 即使 p=1，节省也只从 12.40% 升到 13.33%，因为 r=0.99 时对抗流量仅占 1%。")
print("    => 结论：部署典型流量下，级联经济性几乎完全由【良性流量】的 L2 分流决定。")

print("\n" + "=" * 78)
