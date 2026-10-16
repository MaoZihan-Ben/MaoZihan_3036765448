# -*- coding: utf-8 -*-
"""
复现 Card & Krueger (1994, AER 84(4)):
"Minimum Wages and Employment: A Case Study of the Fast-Food Industry
 in New Jersey and Pennsylvania"

用法（在仓库根目录 Name_StudentID 下执行）：
    python code/run_all.py

本脚本只依赖 data/raw/public.csv（410 家快餐店两轮电话调查数据）。
所有路径基于 __file__ 定位，使用 pathlib，无任何硬编码绝对路径，
因此在本地电脑和 GitHub 上被 AI agent 拉取后均可正常运行。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无界面后端，保证在服务器/CI 上也能出图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# 1. 路径设置（全部由 __file__ 推导，禁止硬编码绝对路径）
# =====================================================================
CODE_DIR = Path(__file__).resolve().parent          # .../Name_StudentID/code
ROOT = CODE_DIR.parent                              # .../Name_StudentID
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_LOGS = ROOT / "outputs" / "logs"

RAW_FILE = DATA_RAW / "public.csv"

# =====================================================================
# 2. 论文设定常量
# =====================================================================
W_OLD = 4.25          # 新泽西州旧的最低工资
W_NEW = 5.05          # 1992-04-01 起新泽西州新最低工资
NJ = 1                # STATE==1 表示新泽西
PA = 0                # STATE==0 表示宾夕法尼亚
CHAIN_NAME = {1: "Burger King", 2: "KFC", 3: "Roy Rogers", 4: "Wendy's"}
PERM_CLOSED = 3       # STATUS2==3 永久关闭
TEMP_CLOSED = [2, 4, 5]  # 装修/道路施工/商场火灾导致的临时关闭

# 论文发表的基准结果（用于最后自动比对），来源：论文各表格
PAPER = {}


def chk(table, item, mine, paper, tol, se_mine=None, se_paper=None):
    """登记一条“本次复现值 vs 论文发表值”的比对记录。"""
    PAPER.setdefault(table, []).append(
        dict(table=table, item=item, mine=mine, paper=paper,
             se_mine=se_mine, se_paper=se_paper, tol=tol)
    )


# =====================================================================
# 3. 基础工具函数
# =====================================================================
def desc(s):
    """返回 (均值, 标准误, 样本量)"""
    s = pd.Series(s).astype("float64").dropna()
    n = int(s.size)
    if n < 2:
        return np.nan, np.nan, n
    return float(s.mean()), float(s.std(ddof=1) / np.sqrt(n)), n


def diff_desc(a, b):
    """返回 (均值差 a-b, 差的联合标准误)"""
    ma, sa, _ = desc(a)
    mb, sb, _ = desc(b)
    return ma - mb, float(np.sqrt(sa ** 2 + sb ** 2))


def fmt(x):
    return "  n/a " if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:6.2f}"


def sep(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# =====================================================================
# 4. 数据遍历 + 清洗
# =====================================================================
def load_raw():
    sep("步骤 0. 数据遍历（data/raw/public.csv）")
    df = pd.read_csv(RAW_FILE)
    print(f"原始数据规模: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"字段: {list(df.columns)}")
    st = df["STATE"].value_counts().to_dict()
    print(f"按州分布  : NJ = {st.get(1, 0)}, PA = {st.get(0, 0)}")
    print("按连锁分布:", {CHAIN_NAME[k]: int(v) for k, v in df["CHAIN"].value_counts().items()})
    print("第二轮状态 STATUS2:", df["STATUS2"].value_counts().to_dict())
    key_vars = ["EMPFT", "EMPPT", "NMGRS", "WAGE_ST", "EMPFT2", "EMPPT2",
                "NMGRS2", "WAGE_ST2", "PSODA", "PFRY", "PENTREE",
                "PSODA2", "PFRY2", "PENTREE2"]
    miss = df[key_vars].isna().sum()
    print("关键变量缺失数:", {k: int(v) for k, v in miss.items() if v > 0})
    return df


def clean(df):
    """
    构造论文所需的分析变量：
      FTE(t) = 全职员工数(含经理) + 0.5 * 兼职员工数
      永久关闭门店的第 2 轮就业记为 0；临时关闭记为缺失
      GAP = 新泽西门店为达到新最低工资所需的工资涨幅，宾州为 0
      全日餐价格 = 中杯汽水 + 小薯条 + 主菜（含税）
      全职员工占比 = 全职员工 / FTE * 100
    """
    d = df.copy()

    # ---- 就业 ----
    d["fte1"] = d["EMPFT"] + d["NMGRS"] + 0.5 * d["EMPPT"]
    d["fte2"] = d["EMPFT2"] + d["NMGRS2"] + 0.5 * d["EMPPT2"]

    closed = d["STATUS2"] == PERM_CLOSED
    temp = d["STATUS2"].isin(TEMP_CLOSED)
    d["closed"] = closed
    d["temp_closed"] = temp

    # 第 2 轮“就业数据可得”：关键分量齐全，或门店已永久关闭（就业=0）
    e2_comp = d[["EMPFT2", "EMPPT2", "NMGRS2"]].notna().all(axis=1)
    d["fte2_avail"] = e2_comp | closed
    d["fte1_avail"] = d[["EMPFT", "EMPPT", "NMGRS"]].notna().all(axis=1)

    # 论文口径：临时关闭门店第 2 轮就业视为缺失
    d.loc[temp, "fte2"] = np.nan
    # 永久关闭门店第 2 轮就业记 0
    d.loc[closed, "fte2"] = 0.0

    # ---- 就业变化 ----
    d["d_fte"] = d["fte2"] - d["fte1"]
    d["pct_fte1"] = d["fte1"].where(d["fte1"] != 0)
    d["pct_change"] = d["d_fte"] / d["pct_fte1"]

    # ---- 最低工资缺口 GAP ----
    d["gap"] = np.where(
        d["STATE"] == NJ,
        np.maximum(0.0, (W_NEW - d["WAGE_ST"]) / d["WAGE_ST"]),
        0.0,
    )

    # ---- 价格 / 其他 ----
    d["meal1"] = d["PSODA"] + d["PFRY"] + d["PENTREE"]
    d["meal2"] = d["PSODA2"] + d["PFRY2"] + d["PENTREE2"]
    d["d_log_meal"] = np.log(d["meal2"]) - np.log(d["meal1"])

    d["pct_ft1"] = d["EMPFT"] / d["fte1"] * 100.0
    d["pct_ft2"] = d["EMPFT2"] / d["fte2"] * 100.0

    d["d_hours"] = d["HRSOPEN2"] - d["HRSOPEN"]
    d["d_regs"] = d["NREGS2"] - d["NREGS"]
    d["d_regs_open"] = d["NREGS112"] - d["NREGS11"]
    d["d_time_raise"] = d["INCTIME2"] - d["INCTIME"]
    d["d_amt_raise"] = d["FIRSTIN2"] - d["FIRSTINC"]

    # ---- 哑变量 ----
    d["nj"] = (d["STATE"] == NJ).astype(int)
    d["bk"] = (d["CHAIN"] == 1).astype(int)
    d["kfc"] = (d["CHAIN"] == 2).astype(int)
    d["rr"] = (d["CHAIN"] == 3).astype(int)
    d["co"] = d["CO_OWNED"].astype(float)

    for c in ["SOUTHJ", "CENTRALJ", "NORTHJ", "PA1", "PA2", "SHORE"]:
        d[c.lower()] = d[c].astype(float)

    # 员工餐项目：1=免费, 2=降价, 3=两者兼有（缺失值保持为缺失，不计入分母）
    for w, mcol in ((1, "MEALS"), (2, "MEALS2")):
        ok = d[mcol].notna()
        d[f"free{w}"] = ((d[mcol] == 1).astype(float) * 100).where(ok)
        d[f"low{w}"] = ((d[mcol] == 2).astype(float) * 100).where(ok)
        d[f"both{w}"] = ((d[mcol] == 3).astype(float) * 100).where(ok)

    # 招聘奖金
    d["bonus1"] = d["BONUS"].astype(float) * 100
    d["bonus2"] = d["SPECIAL2"].astype(float) * 100

    # ---- 论文回归样本（Table 4/5/6：N=357）----
    # 两轮就业可得 且 有起始工资（第 2 轮缺失但门店永久关闭的仍纳入）
    d["s357"] = (d["fte1_avail"] & d["fte2_avail"]
                 & d["WAGE_ST"].notna()
                 & (d["WAGE_ST2"].notna() | d["closed"]))
    # 平衡样本（Table 3 行 4/5）
    d["balanced"] = d["fte1_avail"] & d["fte2_avail"]

    d.to_csv(DATA_PROCESSED / "cleaned_did.csv", index=False)
    print(f"\n清洗完成 -> {DATA_PROCESSED / 'cleaned_did.csv'}")
    print(f"论文 Table 4/5/6 回归样本 (N=357 口径): N = {int(d['s357'].sum())}")
    print(f"平衡样本 (Table 3 行4/5):            N = {int(d['balanced'].sum())}")
    return d


def sub(d, mask_col="s357"):
    return d[d[mask_col]].copy()


# =====================================================================
# 5. Table 2：关键变量的分州均值
# =====================================================================
def table2(d):
    sep("步骤 1. 复现 Table 2 —— 分州均值（Stores in NJ / PA）")
    rows = []

    nj = d["STATE"] == NJ
    pa = d["STATE"] == PA

    # 1. 门店类型分布
    for c, name in CHAIN_NAME.items():
        p1 = (d.loc[nj, "CHAIN"] == c).mean() * 100
        p2 = (d.loc[pa, "CHAIN"] == c).mean() * 100
        rows.append(dict(item=f"1. {name} (%)", NJ=p1, PA=p2, t=None))
    rows.append(dict(item="1. Company-owned (%)",
                     NJ=d.loc[nj, "CO_OWNED"].mean() * 100,
                     PA=d.loc[pa, "CO_OWNED"].mean() * 100, t=None))

    # 2/3. 两轮均值
    specs = [("FTE employment", "fte1", "fte2"),
             ("Percentage full-time employees", "pct_ft1", "pct_ft2"),
             ("Starting wage", "WAGE_ST", "WAGE_ST2"),
             ("Price of full meal", "meal1", "meal2"),
             ("Hours open (weekday)", "HRSOPEN", "HRSOPEN2"),
             ("Recruiting bonus", "bonus1", "bonus2")]
    for name, v1, v2 in specs:
        for wave, v in ((1, v1), (2, v2)):
            m1, s1, _ = desc(d.loc[nj, v])
            m2, s2, _ = desc(d.loc[pa, v])
            t = (m1 - m2) / np.sqrt(s1 ** 2 + s2 ** 2)
            rows.append(dict(item=f"{wave}. {name}", NJ=m1, PA=m2, t=t))

    # 工资恰好等于最低工资的比例
    for wave, wage, base in ((1, "WAGE_ST", W_OLD), (2, "WAGE_ST2", W_OLD)):
        p1 = (d.loc[nj, wage] == base).mean() * 100
        p2 = (d.loc[pa, wage] == base).mean() * 100
        rows.append(dict(item=f"{wave}. Wage = $4.25 (%)", NJ=p1, PA=p2, t=None))
    p1 = (d.loc[nj, "WAGE_ST2"] == W_NEW).mean() * 100
    p2 = (d.loc[pa, "WAGE_ST2"] == W_NEW).mean() * 100
    rows.append(dict(item="2. Wage = $5.05 (%)", NJ=p1, PA=p2, t=None))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "table2_means.csv", index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

    # 与论文发表值登记比对
    paper_t2 = {
        "1. FTE employment": (20.4, 23.3),
        "2. FTE employment": (21.0, 21.2),
        "1. Starting wage": (4.61, 4.63),
        "2. Starting wage": (5.08, 4.62),
        "1. Price of full meal": (3.35, 3.04),
        "2. Price of full meal": (3.41, 3.03),
        "1. Percentage full-time employees": (32.8, 35.0),
        "2. Percentage full-time employees": (35.9, 30.4),
        "1. Hours open (weekday)": (14.4, 14.5),
        "2. Hours open (weekday)": (14.4, 14.7),
        "1. Recruiting bonus": (23.6, 29.1),
        "2. Recruiting bonus": (20.3, 23.4),
        "1. Wage = $4.25 (%)": (30.5, 32.9),
        "2. Wage = $4.25 (%)": (0.0, 25.3),
        "2. Wage = $5.05 (%)": (85.2, 1.3),
    }
    for item, (p_nj, p_pa) in paper_t2.items():
        r = out[out.item == item]
        if len(r):
            tol = 0.5 if "%" in item else 0.15
            chk("Table2", f"{item} (NJ)", float(r.NJ.iloc[0]), p_nj, tol)
            chk("Table2", f"{item} (PA)", float(r.PA.iloc[0]), p_pa, tol)
    return out


# =====================================================================
# 6. Table 3：差分（Difference-in-Differences）
# =====================================================================
def _cell(v):
    m, s, n = desc(v)
    return m, s, n


def table3(d):
    sep("步骤 2. 复现 Table 3 —— 平均就业：最低工资上调前后（DID）")
    nj = d["STATE"] == NJ
    pa = d["STATE"] == PA
    njw = d["WAGE_ST"]

    groups = {
        "(i) PA": pa,
        "(ii) NJ": nj,
        "(iv) wage=$4.25": nj & (njw == W_OLD),
        "(v) wage=$4.26-4.99": nj & (njw > W_OLD) & (njw < 5.0),
        "(vi) wage>=$5.00": nj & (njw >= 5.0),
    }

    def pair(a, b):
        """两组之差：均值差 (a-b) 与标准误"""
        return diff_desc(a, b)

    def get(name, series):
        return series[groups[name]]

    res = {}
    # 行1：上调前就业（全部可得观测）
    s1 = d["fte1"]
    for k in groups:
        res[(1, k)] = _cell(get(k, s1))
    res[(1, "(iii) NJ-PA")] = pair(get("(ii) NJ", s1), get("(i) PA", s1)) + (0,)
    res[(1, "(vii) low-high")] = pair(get("(iv) wage=$4.25", s1), get("(vi) wage>=$5.00", s1)) + (0,)
    res[(1, "(viii) mid-high")] = pair(get("(v) wage=$4.26-4.99", s1), get("(vi) wage>=$5.00", s1)) + (0,)

    # 行2：上调后就业
    s2 = d["fte2"]
    for k in groups:
        res[(2, k)] = _cell(get(k, s2))
    res[(2, "(iii) NJ-PA")] = pair(get("(ii) NJ", s2), get("(i) PA", s2)) + (0,)
    res[(2, "(vii) low-high")] = pair(get("(iv) wage=$4.25", s2), get("(vi) wage>=$5.00", s2)) + (0,)
    res[(2, "(viii) mid-high")] = pair(get("(v) wage=$4.26-4.99", s2), get("(vi) wage>=$5.00", s2)) + (0,)

    # 行3：就业变化（全部可得观测）
    for k in groups:
        m1, _, _ = res[(1, k)]
        m2, _, _ = res[(2, k)]
        res[(3, k)] = (m2 - m1, np.nan, 0)
    res[(3, "(iii) NJ-PA")] = (res[(3, "(ii) NJ")][0] - res[(3, "(i) PA")][0], np.nan, 0)
    res[(3, "(vii) low-high")] = (res[(3, "(iv) wage=$4.25")][0] - res[(3, "(vi) wage>=$5.00")][0], np.nan, 0)
    res[(3, "(viii) mid-high")] = (res[(3, "(v) wage=$4.26-4.99")][0] - res[(3, "(vi) wage>=$5.00")][0], np.nan, 0)

    # 行4/5：平衡样本上的就业变化
    bal = d["balanced"]
    bal5 = d["balanced"] | d["temp_closed"]        # 行5：把临时关闭门店计入（就业记为 0）
    for row, s2v, use in ((4, d["fte2"], bal), (5, d["fte2"].mask(d["temp_closed"], 0.0), bal5)):
        ch = (s2v - d["fte1"])[use]
        for k in groups:
            res[(row, k)] = _cell(ch[groups[k] & use])
        res[(row, "(iii) NJ-PA")] = pair(ch[groups["(ii) NJ"] & use], ch[groups["(i) PA"] & use]) + (0,)
        res[(row, "(vii) low-high")] = pair(ch[groups["(iv) wage=$4.25"] & use], ch[groups["(vi) wage>=$5.00"] & use]) + (0,)
        res[(row, "(viii) mid-high")] = pair(ch[groups["(v) wage=$4.26-4.99"] & use], ch[groups["(vi) wage>=$5.00"] & use]) + (0,)

    cols = ["(i) PA", "(ii) NJ", "(iii) NJ-PA", "(iv) wage=$4.25",
            "(v) wage=$4.26-4.99", "(vi) wage>=$5.00",
            "(vii) low-high", "(viii) mid-high"]
    labels = {1: "1. FTE employment before", 2: "2. FTE employment after",
              3: "3. Change in mean FTE", 4: "4. Change, balanced sample",
              5: "5. Change, temp-closed = 0"}
    out = []
    for row in (1, 2, 3, 4, 5):
        rec = {"Variable": labels[row]}
        for c in cols:
            m, s, _ = res[(row, c)]
            rec[c] = m
            rec[c + " (se)"] = s
        out.append(rec)
    o = pd.DataFrame(out)
    o.to_csv(OUT_TABLES / "table3_did.csv", index=False)
    show = o[[c for c in o.columns if not c.endswith("(se)")]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

    # 登记比对（论文 Table 3）
    paper_t3 = {
        (1, "(ii) NJ"): 20.44, (1, "(i) PA"): 23.33, (1, "(iii) NJ-PA"): -2.89,
        (2, "(ii) NJ"): 21.03, (2, "(i) PA"): 21.17, (2, "(iii) NJ-PA"): -0.14,
        (3, "(ii) NJ"): 0.59, (3, "(i) PA"): -2.16, (3, "(iii) NJ-PA"): 2.76,
        (4, "(ii) NJ"): 0.47, (4, "(i) PA"): -2.28, (4, "(iii) NJ-PA"): 2.75,
        (5, "(iii) NJ-PA"): 2.51,
        (4, "(vi) wage>=$5.00"): -2.16, (4, "(iv) wage=$4.25"): 1.21,
        (4, "(vii) low-high"): 3.36, (4, "(viii) mid-high"): 2.87,
    }
    for (row, c), pv in paper_t3.items():
        chk("Table3", f"[行{row}] {c}", res[(row, c)][0], pv, 0.05)
    return o


# =====================================================================
# 7. Table 4：就业变化的简约式回归
# =====================================================================
def table4(d):
    sep("步骤 3. 复现 Table 4 —— 就业变化的简约式回归模型")
    dd = sub(d)
    ctrl = "bk + kfc + rr + co"
    reg = "northj + centralj + pa1 + pa2"   # 新泽西两地区 + 宾州两地区

    m1 = smf.ols("d_fte ~ nj", dd).fit()
    m2 = smf.ols(f"d_fte ~ nj + {ctrl}", dd).fit()
    m3 = smf.ols("d_fte ~ gap", dd).fit()
    m4 = smf.ols(f"d_fte ~ gap + {ctrl}", dd).fit()
    m5 = smf.ols(f"d_fte ~ gap + {ctrl} + {reg}", dd).fit()

    rows = []
    for name, spec, key in [("(i)", m1, "nj"), ("(ii)", m2, "nj"),
                            ("(iii)", m3, "gap"), ("(iv)", m4, "gap"),
                            ("(v)", m5, "gap")]:
        rows.append(dict(model=name,
                         coef=spec.params[key], se=spec.bse[key],
                         t=spec.tvalues[key], ser=np.sqrt(spec.mse_resid),
                         n=int(spec.nobs)))
    o = pd.DataFrame(rows)
    o.to_csv(OUT_TABLES / "table4_employment_regressions.csv", index=False)
    print(o.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print(f"被解释变量均值 = {dd['d_fte'].mean():.3f}, 标准差 = {dd['d_fte'].std():.3f}")

    for name, spec, key in [("(i)", m1, "nj"), ("(ii)", m2, "nj"),
                            ("(iii)", m3, "gap"), ("(iv)", m4, "gap"),
                            ("(v)", m5, "gap")]:
        pv = {"(i)": 2.33, "(ii)": 2.30, "(iii)": 15.65, "(iv)": 14.92, "(v)": 11.91}[name]
        psev = {"(i)": 1.19, "(ii)": 1.20, "(iii)": 6.08, "(iv)": 6.21, "(v)": 7.39}[name]
        chk("Table4", f"{name} coefficient", spec.params[key], pv, 0.10,
            se_mine=spec.bse[key], se_paper=psev)
    chk("Table4", "N", int(dd.shape[0]), 357, 1)
    chk("Table4", "mean(dep)", dd["d_fte"].mean(), -0.237, 0.02)
    chk("Table4", "sd(dep)", dd["d_fte"].std(), 8.825, 0.05)
    return o


# =====================================================================
# 8. Table 5：稳健性检验
# =====================================================================
def table5(d):
    sep("步骤 4. 复现 Table 5 —— 简约式就业模型的稳健性检验")
    ctrl = "bk + kfc + rr + co"
    reg = "northj + centralj + pa1 + pa2"

    def fit(df, y, key):
        r = smf.ols(f"{y} ~ {key} + {ctrl}", df).fit()
        return r.params[key], r.bse[key]

    # 各种就业口径
    d2 = d.copy()
    d2["fte2b"] = d2["fte2"].mask(d2["temp_closed"], 0.0)
    d2["d_fte_b"] = d2["fte2b"] - d2["fte1"]
    d2["fte1_nomgr"] = d2["EMPFT"] + 0.5 * d2["EMPPT"]
    d2["fte2_nomgr"] = d2["EMPFT2"] + 0.5 * d2["EMPPT2"]
    d2.loc[d2["closed"], "fte2_nomgr"] = 0.0
    d2.loc[d2["temp_closed"], "fte2_nomgr"] = np.nan
    d2["d_fte_nomgr"] = d2["fte2_nomgr"] - d2["fte1_nomgr"]
    for w in (0.4, 0.6):
        f1 = d2["EMPFT"] + d2["NMGRS"] + w * d2["EMPPT"]
        f2 = d2["EMPFT2"] + d2["NMGRS2"] + w * d2["EMPPT2"]
        f2 = f2.where(~d2["closed"], 0.0).where(~d2["temp_closed"], np.nan)
        d2[f"d_fte_w{int(w*10)}"] = f2 - f1

    base = sub(d)
    rows = []

    def record(name, coef, se, key):
        rows.append(dict(specification=name, key=key, coef=coef, se=se))

    c, s = fit(base, "d_fte", "nj");   record("1. Base (NJ dummy)", c, s, "nj")
    c, s = fit(base, "d_fte", "gap");  record("1. Base (Gap)", c, s, "gap")

    nb = sub(d2, "s357")
    nb2 = d2[d2["s357"] | d2["temp_closed"]]        # 行2：把临时关闭门店就业记为 0
    c, s = fit(nb2, "d_fte_b", "nj");   record("2. Temp-closed = 0 (NJ dummy)", c, s, "nj")
    c, s = fit(nb2, "d_fte_b", "gap");  record("2. Temp-closed = 0 (Gap)", c, s, "gap")

    c, s = fit(nb, "d_fte_nomgr", "nj");  record("3. Drop managers (NJ dummy)", c, s, "nj")
    c, s = fit(nb, "d_fte_nomgr", "gap"); record("3. Drop managers (Gap)", c, s, "gap")

    for w, lbl in ((0.4, "4. PT = 0.4xFT"), (0.6, "5. PT = 0.6xFT")):
        wc = f"d_fte_w{int(w*10)}"
        c, s = fit(nb, wc, "nj");  record(f"{lbl} (NJ dummy)", c, s, "nj")
        c, s = fit(nb, wc, "gap"); record(f"{lbl} (Gap)", c, s, "gap")

    no_shore = nb[nb["shore"] == 0]
    c, s = fit(no_shore, "d_fte", "nj");  record("6. Drop NJ shore (NJ dummy)", c, s, "nj")
    c, s = fit(no_shore, "d_fte", "gap"); record("6. Drop NJ shore (Gap)", c, s, "gap")

    few_calls = nb[nb["NCALLS"] <= 2]
    c, s = fit(few_calls, "d_fte", "nj");  record("8. Drop >2 calls (NJ dummy)", c, s, "nj")
    c, s = fit(few_calls, "d_fte", "gap"); record("8. Drop >2 calls (Gap)", c, s, "gap")

    # 比例变化模型
    p = base[base["pct_change"].notna()]
    pc = smf.ols(f"pct_change ~ nj + {ctrl}", p).fit()
    record("1b. Prop. change (NJ dummy)", pc.params["nj"], pc.bse["nj"], "nj")
    pg = smf.ols(f"pct_change ~ gap + {ctrl}", p).fit()
    record("1b. Prop. change (Gap)", pg.params["gap"], pg.bse["gap"], "gap")
    wls = smf.wls(f"pct_change ~ gap + {ctrl}", p, weights=p["fte1"]).fit()
    record("9. Weighted by init. emp. (Gap)", wls.params["gap"], wls.bse["gap"], "gap")
    wls_nj = smf.wls(f"pct_change ~ nj + {ctrl}", p, weights=p["fte1"]).fit()
    record("9. Weighted by init. emp. (NJ dummy)", wls_nj.params["nj"], wls_nj.bse["nj"], "nj")

    # 12. 仅宾州门店：GAP 定义为达到 5.05 所需的涨幅
    pa_only = nb[nb["STATE"] == PA].copy()
    pa_only["gap_pa"] = np.maximum(0.0, (W_NEW - pa_only["WAGE_ST"]) / pa_only["WAGE_ST"])
    r = smf.ols(f"d_fte ~ gap_pa + {ctrl}", pa_only).fit()
    record("12. PA only (Gap)", r.params["gap_pa"], r.bse["gap_pa"], "gap_pa")

    o = pd.DataFrame(rows)
    o.to_csv(OUT_TABLES / "table5_robustness.csv", index=False)
    print(o.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    for name, pv, psev in [("1. Base (NJ dummy)", 2.30, 1.19),
                           ("1. Base (Gap)", 14.92, 6.21),
                           ("2. Temp-closed = 0 (NJ dummy)", 2.20, 1.21),
                           ("2. Temp-closed = 0 (Gap)", 14.42, 6.31),
                           ("3. Drop managers (NJ dummy)", 2.27, 1.17),
                           ("3. Drop managers (Gap)", 14.69, 6.05),
                           ("4. PT = 0.4xFT (NJ dummy)", 2.34, 1.20),
                           ("4. PT = 0.4xFT (Gap)", 15.23, 6.23),
                           ("5. PT = 0.6xFT (NJ dummy)", 2.27, 1.21),
                           ("5. PT = 0.6xFT (Gap)", 14.60, 6.26),
                           ("6. Drop NJ shore (NJ dummy)", 2.58, 1.19),
                           ("6. Drop NJ shore (Gap)", 16.88, 6.36),
                           ("8. Drop >2 calls (NJ dummy)", 2.41, 1.28),
                           ("8. Drop >2 calls (Gap)", 14.08, 7.11),
                           ("1b. Prop. change (NJ dummy)", 0.03, 0.05),
                           ("1b. Prop. change (Gap)", 0.34, 0.26),
                           ("9. Weighted by init. emp. (NJ dummy)", 0.13, 0.05),
                           ("9. Weighted by init. emp. (Gap)", 0.81, 0.26),
                           ("12. PA only (Gap)", -0.30, 22.00)]:
        tol = 0.15 if name.startswith(("1b", "9")) else 0.10
        r_ = o[o.specification == name]
        if len(r_):
            chk("Table5", name, float(r_.coef.iloc[0]), pv, tol,
                se_mine=float(r_.se.iloc[0]), se_paper=psev)
    return o


# =====================================================================
# 9. Table 6：对其他结果的影响（非工资补偿等）
# =====================================================================
def table6(d):
    sep("步骤 5. 复现 Table 6 —— 最低工资对其他结果的影响")
    dd = d.copy()
    dd["d_pct_ft"] = dd["pct_ft2"] - dd["pct_ft1"]
    dd["d_free"] = dd["free2"] - dd["free1"]
    dd["d_low"] = dd["low2"] - dd["low1"]
    dd["d_both"] = dd["both2"] - dd["both1"]
    dd["slope1"] = dd["FIRSTINC"] / dd["INCTIME"]
    dd["slope2"] = dd["FIRSTIN2"] / dd["INCTIME2"]
    dd["d_slope"] = dd["slope2"] - dd["slope1"]
    dd["d_amt"] = dd["FIRSTIN2"] - dd["FIRSTINC"]

    outcomes = [("1. Fraction full-time workers", "d_pct_ft"),
                ("2. Hours open per weekday", "d_hours"),
                ("3. Number of cash registers", "d_regs"),
                ("4. Registers open at 11 AM", "d_regs_open"),
                ("5. Low-price meal program", "d_low"),
                ("6. Free meal program", "d_free"),
                ("7. Combination of meals", "d_both"),
                ("8. Time to first raise", "d_time_raise"),
                ("9. Amount of first raise", "d_amt"),
                ("10. Slope of wage profile", "d_slope")]

    ctrl = "bk + kfc + rr + co"
    reg = "northj + centralj + pa1 + pa2"
    rows = []
    for name, y in outcomes:
        nj = dd.loc[dd["STATE"] == NJ, y]
        pa = dd.loc[dd["STATE"] == PA, y]
        mnj, _, _ = desc(nj)
        mpa, _, _ = desc(pa)
        diff, dse = diff_desc(nj, pa)
        m4 = smf.ols(f"{y} ~ nj + {ctrl}", dd).fit()
        m5 = smf.ols(f"{y} ~ gap + {ctrl}", dd).fit()
        m6 = smf.ols(f"{y} ~ gap + {ctrl} + {reg}", dd).fit()
        rows.append(dict(outcome=name, NJ=mnj, PA=mpa, NJ_PA=diff,
                         nj_dummy=m4.params["nj"], nj_dummy_se=m4.bse["nj"],
                         gap=m5.params["gap"], gap_se=m5.bse["gap"],
                         gap_reg=m6.params["gap"], gap_reg_se=m6.bse["gap"]))
    o = pd.DataFrame(rows)
    o.to_csv(OUT_TABLES / "table6_other_outcomes.csv", index=False)
    print(o[["outcome", "NJ", "PA", "NJ_PA", "nj_dummy", "gap", "gap_reg"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.2f}"))

    for name, pv in [("1. Fraction full-time workers", 7.29),
                     ("2. Hours open per weekday", -0.11),
                     ("3. Number of cash registers", -0.17),
                     ("4. Registers open at 11 AM", 0.17),
                     ("5. Low-price meal program", -3.39),
                     ("6. Free meal program", 2.00),
                     ("7. Combination of meals", 1.09),
                     ("8. Time to first raise", 2.51),
                     ("9. Amount of first raise", 0.01),
                     ("10. Slope of wage profile", -0.01)]:
        r_ = o[o.outcome == name]
        if len(r_):
            chk("Table6", f"{name} (NJ-PA)", float(r_.NJ_PA.iloc[0]), pv, 0.20)
    return o


# =====================================================================
# 10. Table 7：价格效应
# =====================================================================
def table7(d):
    sep("步骤 6. 复现 Table 7 —— 最低工资对全日餐价格的影响")
    dd = sub(d)
    dd = dd[dd["d_log_meal"].notna()].copy()
    ctrl = "bk + kfc + rr + co"
    reg = "northj + centralj + pa1 + pa2"

    m1 = smf.ols("d_log_meal ~ nj", dd).fit()
    m2 = smf.ols(f"d_log_meal ~ nj + {ctrl}", dd).fit()
    m3 = smf.ols("d_log_meal ~ gap", dd).fit()
    m4 = smf.ols(f"d_log_meal ~ gap + {ctrl}", dd).fit()
    m5 = smf.ols(f"d_log_meal ~ gap + {ctrl} + {reg}", dd).fit()

    rows = []
    for name, spec, key in [("(i)", m1, "nj"), ("(ii)", m2, "nj"),
                            ("(iii)", m3, "gap"), ("(iv)", m4, "gap"),
                            ("(v)", m5, "gap")]:
        rows.append(dict(model=name, coef=spec.params[key], se=spec.bse[key],
                         t=spec.tvalues[key], ser=np.sqrt(spec.mse_resid),
                         n=int(spec.nobs)))
    o = pd.DataFrame(rows)
    o.to_csv(OUT_TABLES / "table7_prices.csv", index=False)
    print(o.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print(f"被解释变量均值 = {dd['d_log_meal'].mean():.4f}, 标准差 = {dd['d_log_meal'].std():.4f}")

    for name, spec, key, pv, psev in [
            ("(i)", m1, "nj", 0.033, 0.014), ("(ii)", m2, "nj", 0.037, 0.014),
            ("(iii)", m3, "gap", 0.077, 0.075), ("(iv)", m4, "gap", 0.146, 0.074),
            ("(v)", m5, "gap", 0.063, 0.089)]:
        chk("Table7", f"{name} coefficient", spec.params[key], pv, 0.02,
            se_mine=spec.bse[key], se_paper=psev)
    chk("Table7", "N", int(dd.shape[0]), 315, 3)
    chk("Table7", "mean(dep)", dd["d_log_meal"].mean(), 0.0173, 0.002)
    chk("Table7", "sd(dep)", dd["d_log_meal"].std(), 0.1017, 0.002)
    return o


# =====================================================================
# 11. Figure 1：起薪分布
# =====================================================================
def figure1(d):
    sep("步骤 7. 复现 Figure 1 —— 两州起薪分布（前 / 后）")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (col, title) in zip(axes, [("WAGE_ST", "February 1992"),
                                       ("WAGE_ST2", "November 1992")]):
        bins = np.arange(4.0, 5.6, 0.1)
        ax.hist(d.loc[d["STATE"] == NJ, col].dropna(), bins=bins, alpha=0.6,
                label="New Jersey", color="#3b6ea5")
        ax.hist(d.loc[d["STATE"] == PA, col].dropna(), bins=bins, alpha=0.6,
                label="Pennsylvania", color="#c0504d")
        ax.set_title(title)
        ax.set_xlabel("Starting wage ($/hour)")
        ax.legend()
    axes[0].set_ylabel("Number of stores")
    fig.suptitle("Figure 1 (reproduction): Distribution of starting wage rates")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "figure1_wage_distribution.png", dpi=150)
    plt.close(fig)
    print(f"已保存 -> {OUT_FIGURES / 'figure1_wage_distribution.png'}")


# =====================================================================
# 12. 与论文结果自动比对
# =====================================================================
def compare():
    sep("步骤 8. 本次复现结果 vs 论文发表结果 比对")
    recs = []
    for table, lst in PAPER.items():
        for r in lst:
            mine, paper, tol = r["mine"], r["paper"], r["tol"]
            ok = (not np.isnan(mine)) and abs(mine - paper) <= tol
            recs.append(dict(table=table, item=r["item"],
                             mine=round(float(mine), 4), paper=paper,
                             diff=round(float(mine) - paper, 4),
                             se_mine=None if r["se_mine"] is None else round(float(r["se_mine"]), 4),
                             se_paper=r["se_paper"],
                             tolerance=tol, status="OK" if ok else "CHECK"))
    o = pd.DataFrame(recs)
    o.to_csv(OUT_TABLES / "comparison_with_paper.csv", index=False)

    print(o[["table", "item", "mine", "paper", "diff", "status"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    n_ok = (o.status == "OK").sum()
    print(f"\n合计 {len(o)} 项比对：OK = {n_ok}，需复核 CHECK = {len(o) - n_ok}")
    if (o.status != "OK").any():
        print("\n需复核的项目：")
        print(o[o.status != "OK"][["table", "item", "mine", "paper", "diff", "tolerance"]]
              .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print(f"\n比对明细 -> {OUT_TABLES / 'comparison_with_paper.csv'}")
    return o


# =====================================================================
# 13. 无法复现的部分（数据不足）
# =====================================================================
def not_reproducible():
    sep("步骤 9. 数据不足、无法复现的论文内容")
    items = [
        ("Table 1", "抽样框 473 家门店与各轮回应率/拒访信息",
         "public.csv 只含 410 家成功受访门店，缺少未回应门店名单"),
        ("Table 5 行 10-11", "Newark（070）与 Camden（080）三码邮编子样本",
         "public.csv 无邮编字段，无法识别这两个子样本"),
        ("Table 8", "各州麦当劳门店开设率（1986-1991）",
         "需各州麦当劳门店数、1986 CPS 零售业工资分布、各州 1990 最低工资，均不在数据包内"),
        ("第 VII 节", "CPS 州级就业-人口比（NJ/PA/NY/US，1991-1992）",
         "需 1991-1992 各月 CPS 原始微观数据，不在数据包内"),
        ("第 III-D 节 IV 检验", "以收银机数作第一轮就业工具变量的工具变量回归",
         "NREGS/NREGS11 可用，可做；但论文未报告完整系数表，仅作说明性验证"),
    ]
    print(f"{'位置':<16}{'内容':<44}原因")
    for a, b, c in items:
        print(f"{a:<16}{b:<44}{c}")
    return items


# =====================================================================
# 主流程
# =====================================================================
def main():
    for p in (DATA_PROCESSED, OUT_TABLES, OUT_FIGURES, OUT_LOGS):
        p.mkdir(parents=True, exist_ok=True)

    print("仓库根目录 ROOT      :", ROOT)
    print("原始数据 RAW_FILE    :", RAW_FILE)
    print("输出表格 OUT_TABLES  :", OUT_TABLES)

    df = load_raw()
    d = clean(df)
    table2(d)
    table3(d)
    table4(d)
    table5(d)
    table6(d)
    table7(d)
    figure1(d)
    comp = compare()
    not_reproducible()

    print("\n" + "=" * 72)
    print("复现完成。输出文件：")
    for f in sorted(OUT_TABLES.glob("*.csv")):
        print("  outputs/tables/" + f.name)
    print("  outputs/figures/figure1_wage_distribution.png")
    print("  data/processed/cleaned_did.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
