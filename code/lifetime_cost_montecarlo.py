# -*- coding: utf-8 -*-
"""
糖尿病餘生醫療成本 —— 蒙地卡羅模擬(機率敏感度分析 PSA)
================================================================
對每位病患模擬 N 次「餘生醫療成本」
(以生命表之逐年存活機率加權、累計至生命表終點 105 歲,見 survival_annuity;
 生命表平均餘命 e(x) 僅輸出於 years_horizon 欄供對照),
把三種不確定性一起隨機抽樣,輸出成本的「平均 + 95% 區間 + 分布」:
〔稽核修正 B-15〕原寫「在確定性版(lifetime_cost.py)之上」(重現包內無此檔)與「時界為生命表
  平均餘命」([修正 13] 改為存活加權年金前之舊文),已改寫。

  (1) 併發症狀態不確定性:依分類模型的 P(大血管)、P(小血管) 抽 Bernoulli,
      每次落入 無 / 僅大 / 僅小 / 大+小 四組之一(來自你的模型,非人工假設)。
  (2) 成本參數不確定性:各組年成本以 Gamma 分布抽樣(均值=陳興寶 2003 表 1,
      變異係數 COST_CV 為 PSA 假設,可調;文獻未提供 SE 時的標準作法)。
  (3) 物價校正:2001→2024 的醫療 CPI 校正倍數取自上海統計年鑑逐年連乘之確定值
      (INFLATION_MEAN = 1.3737、INFLATION_CV = 0.0,不再抽樣;年均 +1.39%)。

成本點估計來源(已核對原文表 1):
  陳興寶等(2003)《中国糖尿病杂志》11(4):238-241,上海復旦大學公共衛生學院。

時界依 HORIZON_MODE 決定(預設 "life_table":存活加權至生命表終點)、折現率 DISCOUNT_RATE(健康經濟常用 3%)。

折現乘數([修正 13]):採存活加權年金 a_x = Σ_{t>=1} v^t · l(x+t)/l(x),
  即逐年以「仍存活之機率」加權後折現,而非把平均餘命 e(x) 的點值代入確定性年金公式
  a(e(x)) = (1-(1+r)^-e(x))/r。本資料實測點值代入平均高估 6.2%(偏誤隨年齡放大至 12.9%)。
  〔稽核修正 A-16〕(1) 記號:自 t=1 起算、每年期末給付者為期末年金 a_x;精算上 ä_x 指自
  t=0 起之期初年金,原寫 ä(x) 為記號誤用(計算本身未變)。(2) 歸因:6.2% 並非全來自 Jensen
  不等式(a(T) 對 T 為凹函數)。令 r→0(無 Jensen 效應)時 e(x)/Σ tpx 平均已為 1.031,因 e(x)
  為含半年之完整餘命而本式期末給付;改用與 e(x) 一致之期中給付慣例時,點值代入平均僅
  高估約 2.0%,即 6.2% 中約 2/3 來自給付時點慣例(數值以本程式之生命表與 100 位病患重算)。

象限 95% 區間口徑([修正 15]):每次模擬先將該象限全體加總,再對 N_SIM 條象限總成本
  取 2.5/97.5 百分位、除以人數還原為每人,與母體總計同一邏輯。原版以「各病患自己的
  百分位再取平均」作為象限區間,長條高度與誤差棒並非同一個量的估計與其不確定性,
  區間被誇大 2.1–3.3 倍。

★ 為何只估「餘生前瞻成本」(現在→80),而不估「全病程終身成本」(發病→80)★
  第 2 型糖尿病為成年後罹患(非天生),理論上完整終身成本應自「發病年齡 = 當前年齡 − 罹病時間」
  起算。但要估「發病到現在」這段歷史成本,必須知道病患過去各年度的併發症狀態;本資料為橫斷面、
  無逐年追蹤,若以「當前狀態套用到過去每一年」會高估早期(剛發病時多半尚無併發症),若以「線性
  爬升」補之則等於用未經實證的假設冒充資料。為守住「不以假設冒充資料」的原則,本模型只估算每個
  數字皆有依據(年齡、模型風險、成本、折現)的『餘生前瞻成本』。
  ★ 罹病時間(Duration)並未浪費:它是分類模型中小血管併發症最強的預測因子(OR≈3.24),
    亦即「罹病越久 → 併發症機率越高 → 餘生成本越高」——Duration 透過影響風險預測來影響成本,
    而非拿去乘一段虛構的歷史成本。此為其正確且完全以資料驅動的用法。
  本模型另以「當前預期狀態」外推未來,未建模逐年病程進展(那需 CORE/UKPDS 級微模擬),屬一階近似。

依賴:pandas, numpy, scikit-learn, matplotlib;需與 diabetes_deterioration_pipeline.py 同目錄。
"""


# ---------------------------------------------------------------------------
# [修正 32] Windows 主控台編碼防護。
#   本專案之輸出含 ✓ ≤ − ä ² ≈ 等字元,不在繁體中文 Windows 之預設編碼 cp950 內。
#   當 stdout 是「主控台」時 Python 走 WriteConsoleW,不受影響;但當 stdout 被
#   導向「管線」(例如 verify_all.py 以 capture_output=True 抓取子程序輸出,或
#   使用者自行 `python x.py > log.txt`)時,Python 改用地區編碼 cp950 編碼,
#   即拋出 UnicodeEncodeError 並中止 —— 程式本身沒錯,卻因為印不出一個勾勾而失敗。
#   此處只改「遇到無法編碼之字元時的行為」(改為以 ? 取代),不動編碼本身,
#   故主控台顯示維持正常。
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass
# ---------------------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython(); _NB = True
except NameError:
    matplotlib.use("Agg"); _NB = False
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

_av = {f.name for f in _fm.fontManager.ttflist}
for _f in ["Microsoft JhengHei", "Microsoft YaHei", "PingFang TC", "Heiti TC",
           "Noto Sans CJK TC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]:
    if _f in _av:
        plt.rcParams["font.sans-serif"] = [_f]; break
plt.rcParams["axes.unicode_minus"] = False

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedGroupKFold,
                                     cross_val_predict)   # 〔稽核修正 CODE-26〕刪除未使用之 GroupKFold

from diabetes_deterioration_pipeline import load_data

SEED = 42
rng = np.random.default_rng(SEED)
OUTDIR = "output_cost_mc"
os.makedirs(OUTDIR, exist_ok=True)

# ---- 設定(可調)----
N_SIM = 5000               # 每位病患的模擬次數
COST_TYPE = "direct"       # "direct"=年直接醫療費用;"total"=年總費用
COST_CV = 0.30             # 成本 PSA 變異係數(假設;可調)
# ---- 物價校正:以《上海統計年鑑》醫療保健類 CPI 逐年連乘(2001 → 2024 年幣值)----
# 陳興寶成本為 2001 年幣值。採上海口徑而非全國,係因成本來源(復旦大學)與 CGM 樣本
# 同為上海,口徑一致性較佳。切勿用「人均衛生支出成長」(~13×),因其含使用量/技術
# 成長,會高估「同一項併發症」之價格。
#
# 環比指數(上年=100),來源:《上海統計年鑑》價格章「居民消費價格(分類)指數」
#   2002-2015 取自表 9.2(1991~2015)之「醫療保健和個人用品」大類
#   2016-2018 取自表 9.2(2016~2018)、2019-2020 表 8.2(2018~2020)
#   2021-2022 表 8.2(2021~2022)、2023-2024 表 8.2(2022~2024)
# ★ 口徑斷點:2015 年(含)以前為「醫療保健和個人用品」,2016 年起「個人用品」移出、
#   成為獨立之「醫療保健」大類。此為現有公布資料範圍內的最佳做法,報告需載明。
# ★ 重疊年交叉核對:2018 兩表皆 102.4、2022 兩表皆 102.1,佐證分段串接無誤。
MEDICAL_CPI = {
    2002: 97.6,  2003: 100.0, 2004: 100.0, 2005: 100.3, 2006: 101.1, 2007: 100.2,
    2008: 103.1, 2009: 99.4,  2010: 103.7, 2011: 104.1, 2012: 100.6, 2013: 100.0,
    2014: 100.4, 2015: 99.3,  2016: 109.0, 2017: 106.6, 2018: 102.4, 2019: 103.3,
    2020: 101.2, 2021: 98.9,  2022: 102.1, 2023: 100.2, 2024: 99.2,
}
COST_BASE_YEAR = 2001      # 成本資料的幣值年(非論文出版年 2003)
TARGET_YEAR = 2024         # 校正到哪一年的幣值


def cumulative_inflation(base_year=COST_BASE_YEAR, target_year=TARGET_YEAR):
    """
    累計倍數 = Π_{y=base+1}^{target} CPI[y]/100。
    環比指數 CPI[y] 表示「y 年相對 y-1 年」,故 CPI[2001] 描述的是 2000→2001,
    成本既為 2001 年幣值即不應計入;2001→2024 跨 23 年,對應 2002 至 2024 共 23 項。
    (連乘而非各年漲幅相加,為國家統計局確認之算法。)
    """
    years = list(range(base_year + 1, target_year + 1))
    missing = [y for y in years if y not in MEDICAL_CPI]
    if missing:
        raise ValueError(f"MEDICAL_CPI 缺少年度:{missing}")
    mult = 1.0
    for y in years:
        mult *= MEDICAL_CPI[y] / 100.0
    return mult


INFLATION_MEAN = cumulative_inflation()   # 2001→2024 = 1.3737(年均 +1.39%)
# 逐年指數為統計年鑑公布之確定數值,非待估參數,故不再對其加設分布。
# 口徑選擇(上海 vs 全國)之影響改以敏感度分析呈現。
INFLATION_CV = 0.0
DISCOUNT_RATE = 0.03
HORIZON_AGE = 80           # 僅在 HORIZON_MODE="fixed_age" 時使用

# [修正 1] 原程式 years = max(HORIZON_AGE - age, 1),但資料中有 8 筆年齡 >= 80
#          (83,84,84,85,85,85,92,97),這些人被強制算成「餘生 1 年」,邏輯不成立且
#          嚴重低估(83 歲者原折現乘數 0.971,而以生命表餘命 7 年計為 6.230,差 6.4 倍;
#          [修正 13] 後改用存活加權年金,83 歲男性為 4.951)。
#          改用生命表平均餘命:對年輕病患結果與原作法相近(57 歲原 23 年 vs 餘命約 24 年),
#          同時正確處理高齡者。保留 "fixed_age" 模式可還原原作法作對照。
HORIZON_MODE = "life_table"    # "life_table"(建議) 或 "fixed_age"(原作法)
# 〔稽核修正 B-15〕此處原有一句「平均餘命(年)—— 量級估計,建議以中國/上海官方生命表精確值
#   取代」,為改用官方生命表前之殘留,與下方說明矛盾,已刪除。
# ============================================================================
# 平均餘命(年)—— 官方精算生命表逐歲計算(非模型估計)
# ============================================================================
# 來源：《中國人身保險業經驗生命表(2025)》非養老類業務一表男女表(CL2)，
# 中國精算師協會公告，單位:1/1000。
#   新浪財經轉載：https://finance.sina.com.cn/roll/2025-10-30/doc-infvscvq7834163.shtml
# 選用 CL2(非養老類業務一表)之依據：中國精算師協會歷次生命表使用通知明定
# 「定期壽險、終身壽險、健康保險應採用非養老類業務一表」——本研究為醫療
# 成本估算，性質上對應健康保險用途，故採 CL2，不採用 CL1(養老類業務表，
# 該表因年金購買者之逆選擇而系統性低估死亡率、不適用於一般族群)。
#
# 以下 QX_MALE_2025／QX_FEMALE_2025 為官方逐歲死亡機率(qx，0~105歲，共106筆，
# 逐字轉錄自官方公告)，平均餘命 e(x) 以標準生命表公式(蔣氏法同款邏輯)由 qx
# 逐步推算而得，非另行估計或校準——每一步(l/L/T/e)均為教科書標準定義，
# 可逐行覆核。
QX_MALE_2025 = [0.319,0.253,0.201,0.160,0.130,0.110,0.099,0.096,0.100,0.111,
    0.129,0.151,0.176,0.202,0.228,0.252,0.274,0.294,0.312,0.329,
    0.347,0.366,0.387,0.410,0.435,0.460,0.486,0.512,0.537,0.562,
    0.587,0.615,0.647,0.685,0.730,0.784,0.847,0.921,1.004,1.097,
    1.202,1.318,1.448,1.593,1.755,1.933,2.129,2.342,2.572,2.822,
    3.093,3.390,3.719,4.087,4.499,4.959,5.468,6.024,6.626,7.269,
    7.952,8.679,9.456,10.298,11.222,12.252,13.414,14.739,16.257,18.005,
    20.019,22.340,25.013,28.086,31.617,35.667,40.307,45.612,51.667,58.563,
    66.396,75.666,84.166,93.467,103.608,114.625,126.542,139.375,153.129,167.791,
    183.334,199.714,216.867,234.710,253.142,272.048,291.295,310.743,330.242,349.642,
    368.796,387.564,405.815,423.437,440.330,1000.000]
QX_FEMALE_2025 = [0.293,0.213,0.156,0.117,0.092,0.078,0.072,0.073,0.078,0.087,
    0.098,0.111,0.123,0.135,0.146,0.154,0.160,0.165,0.169,0.172,
    0.175,0.178,0.182,0.187,0.192,0.199,0.205,0.212,0.219,0.227,
    0.235,0.243,0.252,0.263,0.276,0.292,0.311,0.335,0.364,0.399,
    0.440,0.487,0.541,0.602,0.669,0.744,0.826,0.916,1.015,1.123,
    1.241,1.370,1.511,1.665,1.835,2.023,2.232,2.463,2.719,3.003,
    3.317,3.666,4.055,4.493,4.992,5.565,6.230,7.008,7.925,9.012,
    10.302,11.836,13.657,15.815,18.361,21.350,24.842,28.894,33.567,38.924,
    45.027,51.700,58.829,66.833,75.787,85.767,96.841,109.070,122.502,137.169,
    153.082,170.224,188.551,207.984,228.411,249.686,271.632,294.044,316.698,339.360,
    361.794,383.773,405.089,425.557,445.027,1000.000]

def _life_table_ex(qx_permille, return_l=False):
    """標準生命表計算:qx(千分位)→ l(x)→ L(x)→ T(x)→ e(x)。
    收尾假設:105歲組 q=1.0(該年齡組必於當年死亡)，L(105) 採平均存活半年之
    標準精算假設。
    [修正 13] 新增 return_l:同時回傳生存人數 l(x)。原版算出 l 後即丟棄,
              使存活加權年金無從計算(見 survival_annuity)。"""
    qx = np.array(qx_permille) / 1000.0
    n = len(qx)
    l = np.empty(n + 1); l[0] = 100000.0
    for x in range(n):
        l[x + 1] = l[x] * (1 - qx[x])
    L = np.empty(n)
    L[:n-1] = (l[:n-1] + l[1:n]) / 2
    L[n-1] = l[n-1] * 0.5
    T = np.cumsum(L[::-1])[::-1]
    ex = T / l[:n]
    return (ex, l) if return_l else ex

_ex_male, _l_male = _life_table_ex(QX_MALE_2025, return_l=True)
_ex_female, _l_female = _life_table_ex(QX_FEMALE_2025, return_l=True)
LIFE_EXPECTANCY_MALE = {a: round(float(_ex_male[a]), 2) for a in range(106)}
LIFE_EXPECTANCY_FEMALE = {a: round(float(_ex_female[a]), 2) for a in range(106)}
# 未知性別時之退回值(兩表之平均，非另一組獨立數據)
LIFE_EXPECTANCY = {a: round((LIFE_EXPECTANCY_MALE[a] + LIFE_EXPECTANCY_FEMALE[a]) / 2, 2)
                   for a in LIFE_EXPECTANCY_MALE}

# ---- [修正 13] 存活加權年金因子 ----------------------------------------
# 原版以 a(e(x)) = (1-(1+r)^-e(x))/r 計算,即「把平均餘命的點值代入確定性年金
# 公式」。a(T) 對 T 二階導數 -(1+r)^(-T)(ln(1+r))^2/r < 0,為嚴格凹函數,
# 依 Jensen 不等式 a(E[T]) > E[a(T)],點值代入必為高估。
# 更根本的是:要估「預期折現餘生成本」,正確乘數應為存活加權年金
#     a_x = Σ_{t>=1} v^t · tpx ,   tpx = l(x+t) / l(x)
# 因為第 t 年只有在病患仍存活時才發生成本。實測(本資料 100 位病患、r=3%):
#     a(e(x)) / a_x 平均 1.0616 → 原版平均高估 6.2%;
#     50 歲 +4.7%、70 歲 +7.8%、83 歲 +12.9%(偏誤隨年齡單調放大;數字為男性表)。
# 由於年齡與大血管併發症風險正相關,此偏誤並非白噪音,而是方向性地
# 偏向高齡與 A/B 象限。
# 〔稽核修正 A-16〕記號:本式自 t=1 起、期末給付,為期末年金 a_x(原寫 ä(x);ä 慣指自 t=0
#   起之期初年金)。歸因:上述 6.2% 中約 2/3 來自給付時點慣例而非 Jensen 效應——令 r→0 時
#   e(x)/Σ tpx 平均即為 1.031;改以期中慣例 Σ_{t>=0} v^(t+0.5)·(tpx+t+1px)/2 為分母時,
#   a(e(x)) 平均僅高估 1.020。以上只改記號與說明,計算未變。

def _surv_annuity_table(l_arr, r):
    """由生存人數 l(x) 建立各整數年齡之存活加權年金因子 a_x=Σ_{t>=1} v^t·l(x+t)/l(x)。
    〔稽核修正 A-16〕原 docstring 寫 ä(x);自 t=1 起算者為期末年金 a_x。"""
    n = len(l_arr) - 1                      # 年齡 0..n-1
    v = 1.0 / (1.0 + r)
    out = np.zeros(n)
    for x in range(n):
        lx = l_arr[x]
        if lx <= 0:
            out[x] = 0.0
            continue
        t = np.arange(1, n + 1 - x)
        if len(t) == 0:
            out[x] = 0.0
            continue
        out[x] = float(np.sum((v ** t) * (l_arr[x + t] / lx)))
    return out


_ann_male = _surv_annuity_table(_l_male, DISCOUNT_RATE)
_ann_female = _surv_annuity_table(_l_female, DISCOUNT_RATE)
_ann_avg = (_ann_male + _ann_female) / 2.0


def survival_annuity(age, gender=None):
    """回傳該年齡、該性別之存活加權年金因子。gender: 0=女、1=男、None=男女平均表。
    非整數年齡以線性內插(與 remaining_years 之處理一致)。"""
    tab = (_ann_female if gender == 0 else
           _ann_male if gender == 1 else _ann_avg)
    return float(np.interp(age, np.arange(len(tab)), tab))


RISK_TABLE = "output_risk/patient_risk_table.csv"   # 象限歸屬之來源(必要;見 _require_risk_table)
QUADRANT_OUTPUTS = [f"{OUTDIR}/quadrant_cost_interval.csv", f"{OUTDIR}/mc_cost_by_quadrant.png"]


def _require_risk_table(record_id):
    """〔稽核修正 B-06〕原本 RISK_TABLE 不存在時,象限欄不產生、象限成本整段
    (quadrant_cost_interval.csv、mc_cost_by_quadrant.png)被靜默略過,程式仍以 0 結束;
    若目錄留有前次之 quadrant_cost_interval.csv,make_figures_v2.py 與驗證腳本會直接讀到
    舊檔。現改為:找不到風險表、或有紀錄查不到象限時,印出明確錯誤、刪除目錄中前次留下
    之象限成本輸出,並以結束碼 1 中止(於蒙地卡羅之前檢查,不浪費運算)。
    回傳 record → quadrant 之對應(與 record_id 同序之 list)。"""
    def _fail(msg):
        gone = [p for p in QUADRANT_OUTPUTS if os.path.exists(p)]
        for p in gone:
            os.remove(p)
        print(f"[錯誤] {msg}")
        print("       象限成本(quadrant_cost_interval.csv、mc_cost_by_quadrant.png)需要各紀錄之"
              "風險象限,請先執行 deterioration_risk.py(run_all.py 已依此順序執行)。")
        if gone:
            print(f"       已刪除前次執行留下之象限成本輸出(避免被誤當本次結果):{', '.join(gone)}")
        sys.exit(1)
    if not os.path.exists(RISK_TABLE):
        _fail(f"找不到 {RISK_TABLE}。")
    _rt = pd.read_csv(RISK_TABLE)[["record", "quadrant"]]
    _map = dict(zip(_rt["record"].astype(str), _rt["quadrant"]))
    q = [_map.get(str(x)) for x in record_id]
    miss = [str(x) for x, v in zip(record_id, q) if v is None or (isinstance(v, float) and np.isnan(v))]
    if miss:
        _fail(f"{RISK_TABLE} 中查不到 {len(miss)} 筆紀錄之象限(例:{', '.join(miss[:3])});"
              f"風險表可能是舊資料或其他版本之輸出。")
    return q

# ---- 併發症機率的類別權重設定(基準情境 vs 敏感度情境)----
# 分類模型以 class_weight="balanced" 配適,可提升少數類的召回、對排序(AUC)有利;
# 但該設定會提高少數類權重、上移截距,使預測機率系統性高於實際盛行率。
# 本成本模型以 rng.random() < p 抽 Bernoulli,吃的是「機率絕對值」而非排序,
# 故此偏誤會直接傳遞至成本。實測(本資料;見 cost_sensitivity_class_weight.csv):
#   大血管 實際盛行率 36.7% / balanced 預測均值 42.1%(+5.4 pp)
#   小血管 實際盛行率 24.8% / balanced 預測均值 33.1%(+8.4 pp)
#   〔稽核修正 B-15〕原寫 43.4%(+6.7 pp)/34.1%(+9.4 pp),為改用 100 次重複 OOF 平均前之
#   舊值;上列為現行輸出(42.1049%、33.1224%)。
# 因此本研究以 balanced 為基準情境(與報告 4.3、4.9 之分類模型一致),
# 另以未加權模型作為敏感度情境,呈現校準差異對成本估計的影響區間。
CLASS_WEIGHT = "balanced"          # 基準情境;敏感度情境為 None
CLASS_WEIGHT_SCENARIOS = ["balanced", None]
# 註:第 2 型糖尿病為成年後才罹患(非天生),故成本自病患「當前年齡」向前累計,
#     屬「餘生前瞻醫療成本」,而非自出生起的終身總額。

# ---- 陳興寶(2003)表 1 四組年成本(2001 RMB,已核對)----
COST = {
    "direct": {"none": 3726.36, "macro": 15373.87, "micro": 11842.05, "both": 38580.24},
    "total":  {"none": 4773.83, "macro": 18836.98, "micro": 15101.63, "both": 44465.52},
}


N_REPEATS = 100   # [修正 A] 折分配重複次數


def oof_prob(X, y, groups, class_weight="__default__"):
    """out-of-fold 陽性機率。class_weight 未指定時採 CLASS_WEIGHT(基準情境)。

    [修正 14] 原簽章之預設值為 None,而函式內以 class_weight == "__default__"
    判斷是否採用 CLASS_WEIGHT,兩者永遠不相等,故 docstring 所述之預設行為
    從未生效(未指定時實際得到未加權模型)。現行四個呼叫點皆顯式傳值,故無
    實害;此處改正預設值使行為與說明一致。

    [修正 A] 原本以 GroupKFold(5)(未設 shuffle)取單一組 out-of-fold 機率,
    折分配不可跨平台重現(〔稽核修正 B-15〕原寫「GroupKFold 不接受 random_state」,
    scikit-learn 1.6 起已支援);而本模型以 rng.random() < p 抽 Bernoulli,
    吃的是機率絕對值,折分配的隨機性會直接傳遞到成本估計。改為
    StratifiedGroupKFold(shuffle=True, random_state=SEED+r) 重複 N_REPEATS 次取平均。"""
    cw = CLASS_WEIGHT if class_weight == "__default__" else class_weight
    lr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=5000, class_weight=cw,
                                              solver="liblinear", random_state=SEED))])
    y = np.asarray(y).astype(int)
    P = np.empty((N_REPEATS, len(y)))
    for r in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)
        P[r] = cross_val_predict(lr, X, y, cv=cv, groups=groups,
                                 method="predict_proba")[:, 1]
    return P.mean(axis=0)


def simulate_costs(p_macro, p_micro, age, is_first, gender=None, seed=SEED, rg=None):
    """給定併發症機率,執行蒙地卡羅並回傳(每人成本平均陣列, 母體總成本 N_SIM 條)。
    [修正 12] 新增 gender 參數,與 main() 的餘命查表口徑一致(先前一律用無性別表)。

    [修正 C] 原本本函式固定自建 np.random.default_rng(SEED),而 main() 另用模組層級
    的 rng(且已被先前的抽樣推進),導致同一次執行對「balanced 基準情境」輸出兩個
    不同的每人平均成本(實測 311,380 與 300,524,差 3.6%),報告引用何者無從判斷。
    改為可由呼叫端傳入同一個 Generator;未傳入時才自建,以維持向後相容。"""
    rg = np.random.default_rng(seed) if rg is None else rg
    c = COST[COST_TYPE]; r = DISCOUNT_RATE

    def gam(mean, cv, size):
        if cv <= 0:
            return np.full(size, mean, dtype=float)
        return rg.gamma(1.0 / cv ** 2, mean * cv ** 2, size=size)

    cost_draw = {k: gam(v, COST_CV, N_SIM) for k, v in c.items()}
    infl_draw = gam(INFLATION_MEAN, INFLATION_CV, N_SIM)
    pop = np.zeros(N_SIM); per = []
    for i in range(len(p_macro)):
        # [修正 13] 改用存活加權年金因子(原為 a(e(x)) 點值代入)
        ann = survival_annuity(age[i], gender[i] if gender is not None else None)
        mac = rg.random(N_SIM) < p_macro[i]
        mic = rg.random(N_SIM) < p_micro[i]
        cs = np.select([mac & mic, mac & ~mic, ~mac & mic],
                       [cost_draw["both"], cost_draw["macro"], cost_draw["micro"]],
                       default=cost_draw["none"])
        lt = cs * infl_draw * ann
        per.append(lt.mean())
        if is_first[i]:
            pop += lt
    return np.array(per), pop


def sensitivity_class_weight(d, targets, pid, age, is_first, gender=None, rg=None):
    """
    敏感度分析:比較 class_weight="balanced"(基準)與 None(未加權)兩種機率設定
    對餘生成本估計的影響,並同時報告兩者的校準品質。
    輸出 cost_sensitivity_class_weight.csv。
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss
    y_mac = targets["Macrovascular"].astype(int).values
    y_mic = targets["Microvascular"].astype(int).values
    rows = []
    for cw in CLASS_WEIGHT_SCENARIOS:
        pm = oof_prob(d, y_mac, pid, class_weight=cw)
        pi = oof_prob(d, y_mic, pid, class_weight=cw)
        per, pop = simulate_costs(pm, pi, age, is_first, gender=gender, rg=rg)
        rows.append(dict(
            class_weight=str(cw),
            macro_prev=y_mac.mean() * 100, macro_pred=pm.mean() * 100,
            micro_prev=y_mic.mean() * 100, micro_pred=pi.mean() * 100,
            macro_auc=roc_auc_score(y_mac, pm), micro_auc=roc_auc_score(y_mic, pi),
            macro_brier=brier_score_loss(y_mac, pm), micro_brier=brier_score_loss(y_mic, pi),
            # [修正 C] 原欄位名 cost_mean 為「全部 109 筆紀錄」之平均,
            #          而 main() 印出的「每人餘生成本平均」為「僅首筆、100 位病患」,
            #          兩者相差約 3.6%,卻同樣被稱為「每人平均」。改以欄名區分,
            #          並同時輸出兩種分母,報告引用時必須指明採用何者。
            cost_mean_per_record=per.mean(),
            cost_median_per_record=float(np.median(per)),
            cost_mean_first_only=float(np.asarray(per)[np.asarray(is_first)].mean()),
            cost_median_first_only=float(np.median(np.asarray(per)[np.asarray(is_first)])),
            pop_total=pop.mean(),
            pop_lo=float(np.percentile(pop, 2.5)), pop_hi=float(np.percentile(pop, 97.5))))
    S = pd.DataFrame(rows)
    # 〔稽核修正 A-19〕原為 S.round(4):小血管 AUC 0.8315266… 被存成 0.8315,恰落在三位小數之
    #   進位邊界,第 2、4 層只能列為「▲ 邊界」而無法判定報告表 4-9-4 之 0.832。改寫全精度
    #   (與其他 CSV 一致),只在報告端進位一次。
    S.to_csv(f"{OUTDIR}/cost_sensitivity_class_weight.csv",
             index=False, encoding="utf-8-sig")

    print("\n=== 敏感度分析:併發症機率之類別權重設定 ===")
    print(f"{'設定':<16}{'大血管預測':>11}{'小血管預測':>11}{'AUC(大/小)':>14}"
          f"{'每人平均*':>11}{'母體總計':>11}")
    print("-" * 76)
    print(f"{'實際盛行率':<14}{S.macro_prev[0]:>10.1f}%{S.micro_prev[0]:>10.1f}%"
          f"{'—':>14}{'—':>11}{'—':>11}")
    for _, x in S.iterrows():
        lab = "balanced(基準)" if x.class_weight == "balanced" else "未加權(敏感度)"
        print(f"{lab:<14}{x.macro_pred:>10.1f}%{x.micro_pred:>10.1f}%"
              f"{x.macro_auc:>7.3f}/{x.micro_auc:.3f}{x.cost_mean_first_only:>11,.0f}"
              f"{x.pop_total/1e4:>9,.0f}萬")
    b, u = S.iloc[0], S.iloc[1]
    print(f"\n  兩情境差距:每人平均 {(u.cost_mean_first_only/b.cost_mean_first_only-1)*100:+.1f}%、"
          f"母體總計 {(u.pop_total/b.pop_total-1)*100:+.1f}%")
    print(f"  校準品質(Brier,越低越好):balanced {b.macro_brier:.4f}/{b.micro_brier:.4f}、"
          f"未加權 {u.macro_brier:.4f}/{u.micro_brier:.4f}")
    print("  * 每人平均 = 僅計每位病患首筆(100 位),與 main() 印出者同一口徑;")
    print("    CSV 另存 cost_mean_per_record(109 筆紀錄之平均,約低 3.6%)。")
    print("  ★ 未加權模型之預測均值較接近實際盛行率、Brier 較低,故基準情境之成本估計")
    print("    可視為上界;兩情境的象限相對倍數與集中度指標不受影響。")
    return S


def gamma_samples(mean, cv, size):
    """以指定均值與變異係數抽 Gamma(保證正值)。cv=0 → 回傳常數。"""
    if cv <= 0:
        return np.full(size, mean, dtype=float)
    k = 1.0 / (cv ** 2)
    theta = mean * (cv ** 2)
    return rng.gamma(k, theta, size=size)


def horizon_label():
    """[修正 1 配套] 圖表與訊息一律用這個字串,避免標題寫死「至 80 歲」與實際時界不符。"""
    return "生命表平均餘命" if HORIZON_MODE == "life_table" else f"至 {HORIZON_AGE} 歲"


def money_label():
    """金額幣值年說明。"""
    return f"{TARGET_YEAR} 年人民幣"


def remaining_years(age, gender=None):
    """依 HORIZON_MODE 回傳成本累計年數。gender: 0=女性、1=男性(與 load_data 編碼一致)、
    None=未知時取男女平均表。"""
    if HORIZON_MODE == "fixed_age":
        years = HORIZON_AGE - age
        if years <= 0:
            print(f"[警告] fixed_age 模式:年齡 {age} 已超過設定終點 {HORIZON_AGE} 歲,"
                  f"「至{HORIZON_AGE}歲」之時界定義已不適用該病患,此模式僅供方法對照,"
                  f"正式結果請使用 HORIZON_MODE='life_table'。")
        return max(years, 1)
    table = (LIFE_EXPECTANCY_FEMALE if gender == 0 else
             LIFE_EXPECTANCY_MALE if gender == 1 else LIFE_EXPECTANCY)
    ages = np.array(sorted(table))
    vals = np.array([table[a] for a in ages])
    return float(np.interp(age, ages, vals))


def main():
    d, med, targets, pid, _ = load_data()   # [修正] load_data 現多回傳 d_raw
    raw = pd.read_excel("Shanghai_T2DM_Summary.xlsx", sheet_name="T2DM")
    record_id = raw["Patient Number"].astype(str).values
    quadrant_of = _require_risk_table(record_id)   # 〔稽核修正 B-06〕缺風險表即中止,不再靜默略過
    age = d["Age"].fillna(d["Age"].median()).values
    gender = d["Gender"].values   # [修正 12] 0=女性、1=男性(與 load_data 編碼一致)；
                                    # 原本完全沒用上,餘命一律查無性別之單一表

    p_macro = oof_prob(d, targets["Macrovascular"].astype(int), pid, class_weight=CLASS_WEIGHT)
    p_micro = oof_prob(d, targets["Microvascular"].astype(int), pid, class_weight=CLASS_WEIGHT)
    c = COST[COST_TYPE]
    r = DISCOUNT_RATE

    # [修正 3] 年成本與物價倍數屬「母體參數不確定性」——其真值對全體病患只有一個,
    #          不會因人而異。原程式在病患迴圈內各自抽樣,109 次獨立抽樣觸發大數法則、
    #          誤差互相抵消,使母體區間相對寬度由 59.5% 縮成 5.6%(實測),
    #          母體 95% 區間因而嚴重低估。改為迴圈外各抽 N_SIM 條、每次模擬全體共用。
    #          併發症狀態(Bernoulli)維持在迴圈內,那才是真正的個體層級不確定性。
    cost_draw = {k: gamma_samples(v, COST_CV, N_SIM) for k, v in c.items()}
    infl_draw = gamma_samples(INFLATION_MEAN, INFLATION_CV, N_SIM)

    # [修正 2] 母體加總只計入每位病患的首筆紀錄。原程式逐 109 筆累加,
    #          但資料為 109 筆紀錄 / 100 位病患(8 位共 9 筆重複回診),
    #          同一病患的多次回診被重複計入母體總成本。個人估計仍全數輸出。
    # 〔稽核修正 CODE-26〕(僅註明,未改演算法)「首筆」= 摘要檔列序中該病患之第一列,
    #   即檔名序號 _0 之紀錄;2017 與 2055 之 _0 並非最早一次就診(其 _1 日期較早)。
    is_first = ~pd.Series(pid.values).duplicated(keep="first").values

    # [修正 15] 象限歸屬原本在迴圈結束後才 merge 進來,迴圈內無從取得。
    #           record→quadrant 對應已於 main() 開頭由 _require_risk_table() 讀入
    #           (〔稽核修正 B-06〕原為「檔案存在才讀,否則 quadrant_of=None 靜默略過」)。

    rows = []
    all_pop = np.zeros(N_SIM)                 # 母體每次模擬的總成本(供母體分布)
    # [修正 15] 象限層級也累加成 N_SIM 條「該象限總成本」,供正確口徑之區間。
    #           原版以「各病患自己的 2.5/97.5 百分位再取平均」作為象限區間,
    #           百分位為非線性算子,mean(percentile_i) != percentile(mean_i):
    #           長條高度是象限平均,誤差棒卻是個人區間之平均,兩者不是同一個量
    #           的估計與其不確定性。實測寬度被誇大 2.1–3.3 倍。
    quad_pop = {}                             # quadrant -> N_SIM 條總成本
    quad_n = {}
    for i in range(len(d)):
        years = remaining_years(age[i], gender[i])          # [修正 12] 傳入性別
        # [修正 13] 年金因子改為存活加權 Σ v^t·tpx。years(=e(x))仍輸出於
        #           years_horizon 欄供對照,但不再用於折現。
        annuity = survival_annuity(age[i], gender[i]) if r > 0 else years

        # (1) 併發症狀態:依模型機率抽 Bernoulli
        mac = rng.random(N_SIM) < p_macro[i]
        mic = rng.random(N_SIM) < p_micro[i]
        # 每個狀態抽該組成本(Gamma),再依抽到的狀態選取
        cost_state = np.select(
            [mac & mic, mac & ~mic, ~mac & mic],
            [cost_draw["both"], cost_draw["macro"], cost_draw["micro"]],
            default=cost_draw["none"])                    # [修正 3] 全體共用抽樣

        lifetime = cost_state * infl_draw * annuity       # N_SIM 條餘生成本
        if is_first[i]:                                   # [修正 2] 只計首筆
            all_pop += lifetime
            q_i = quadrant_of[i] if quadrant_of is not None else None   # [修正 15]
            if q_i is not None and not (isinstance(q_i, float) and np.isnan(q_i)):
                if q_i not in quad_pop:
                    quad_pop[q_i] = np.zeros(N_SIM); quad_n[q_i] = 0
                quad_pop[q_i] += lifetime; quad_n[q_i] += 1
        rows.append(dict(
            record=record_id[i], age=int(round(age[i])),
            years_horizon=round(years, 1),
            annuity_factor=round(float(annuity), 4),   # [修正 13] 存活加權年金因子
            counted_in_population=bool(is_first[i]),
            P_macro=round(float(p_macro[i]), 3), P_micro=round(float(p_micro[i]), 3),
            lifetime_mean=float(lifetime.mean()),
            lifetime_lo=float(np.percentile(lifetime, 2.5)),
            lifetime_hi=float(np.percentile(lifetime, 97.5)),
            annual_mean=float((cost_state * infl_draw).mean()),
        ))

    out = pd.DataFrame(rows)
    if os.path.exists(RISK_TABLE):
        rt = pd.read_csv(RISK_TABLE)[["record", "quadrant", "long_group"]]
        out = out.merge(rt, on="record", how="left")
    out_round = out.copy()
    for col in ["lifetime_mean", "lifetime_lo", "lifetime_hi", "annual_mean"]:
        out_round[col] = out_round[col].round(0)
    out_round.to_csv(f"{OUTDIR}/patient_lifetime_cost_mc.csv", index=False, encoding="utf-8-sig")

    # [修正 29] 生命表之 e(x) 與年金因子 a_x(〔稽核修正 A-16〕原寫 ä(x))原本只存在於程式記憶體中,
    #   決策 App 卻把 lifeMale / lifeFemale 逐歲值寫死在 JS 常數裡。
    #   沒有任何輸出檔載有這些數字,四層驗證因此無從覆核 App 的餘命常數。
    _ages = np.arange(len(_ex_male))
    pd.DataFrame({
        "age": _ages,
        "ex_male": np.round(_ex_male, 4), "ex_female": np.round(_ex_female, 4),
        "ex_avg": np.round((_ex_male + _ex_female) / 2, 4),
        "annuity_male": np.round(_ann_male, 4), "annuity_female": np.round(_ann_female, 4),
        "annuity_avg": np.round(_ann_avg, 4),
    }).to_csv(f"{OUTDIR}/life_table_ex.csv", index=False, encoding="utf-8-sig")
    print(f"生命表 e(x) 與存活加權年金 a_x(0–{len(_ages)-1} 歲、分性別) → {OUTDIR}/life_table_ex.csv")   # 〔稽核修正 A-16〕

    unit = "年直接醫療費用" if COST_TYPE == "direct" else "年總費用"
    print(f"蒙地卡羅:N={N_SIM}/人;成本源 陳興寶 2003 表 1({unit},2001 RMB)")
    _n = TARGET_YEAR - COST_BASE_YEAR
    print(f"物價校正={INFLATION_MEAN:.4f}(上海醫療保健類 CPI {COST_BASE_YEAR}→{TARGET_YEAR} "
          f"逐年連乘 {_n} 項,年均 {INFLATION_MEAN ** (1/_n) - 1:+.2%})")
    print(f"成本CV={COST_CV}、折現 {r:.0%}、時界={'生命表餘命' if HORIZON_MODE=='life_table' else f'至 {HORIZON_AGE} 歲'}\n")
    print(f"每人餘生成本平均({horizon_label()},{money_label()}):  mean=%.0f  median=%.0f"
          "  [僅計首筆,與母體總計、圖2、圖4口徑一致]" %
          (out.loc[out["counted_in_population"], "lifetime_mean"].mean(),
           out.loc[out["counted_in_population"], "lifetime_mean"].median()))
    print(f"母體餘生總成本({horizon_label()},{money_label()}):    mean=%.0f  95%%模擬區間=[%.0f, %.0f]" %
          (all_pop.mean(), np.percentile(all_pop, 2.5), np.percentile(all_pop, 97.5)))

    # ---- 圖 1:各風險象限的終身成本(平均 + 95% 區間誤差棒)----
    # [修正 8] 原僅依 out.groupby(...)、未濾 counted_in_population,重複回診病患
    # 之估計會被計入兩次,使象限平均被同一人的多筆紀錄拉偏(A 象限原被拉低約 2.7 萬)。
    # 與圖 2、圖 4、母體總計、印出之每人平均改為同一口徑:僅計每位病患首筆。
    if "quadrant" in out.columns:
        # [修正 15] 區間口徑改正。原版:lo/hi = 各病患自己的 2.5/97.5 百分位再取平均,
        #   即 mean_i(percentile(lifetime_i))。長條高度估的是「象限平均成本」,
        #   誤差棒卻是「個人成本之離散度的平均」,兩者不是同一個量。百分位為非線性
        #   算子,mean(percentile_i) != percentile(mean_i),且個體 Bernoulli 在平均
        #   過程中並不會互相抵消,故區間被系統性誇大(以現行 quadrant_cost_interval.csv
        #   之 (hi_percap−lo_percap)/(hi−lo) 計為 A 2.20、B 2.10、C 3.07、D 3.25 倍;
        #   〔稽核修正 B-15〕原寫 A 2.27、B 2.14、C 3.10、D 3.26 為舊版數字)。
        #   此誇大使報告誤判「各象限區間均甚寬且彼此重疊」。
        #   正確做法:每次模擬先把該象限全體加總,再對這 N_SIM 條象限總成本取百分位,
        #   最後除以人數還原為每人。與母體總計之 all_pop 完全同一邏輯。
        g = out[out["counted_in_population"]].groupby("quadrant").agg(
            n=("record", "size"),
            mean=("lifetime_mean", "mean"),
            lo_percap=("lifetime_lo", "mean"),      # 保留舊口徑供對照,不再用於作圖
            hi_percap=("lifetime_hi", "mean")).reindex(
            ["A 立即介入(長高+短高)", "B 慢性追蹤(長高+短低)",
             "C 血糖波動注意(長低+短高)", "D 常規追蹤(長低+短低)"]).dropna()
        _lo, _hi = [], []
        for q in g.index:
            if q in quad_pop and quad_n.get(q):
                a = quad_pop[q] / quad_n[q]                   # 每人平均之模擬分布
                _lo.append(float(np.percentile(a, 2.5)))
                _hi.append(float(np.percentile(a, 97.5)))
            else:
                _lo.append(float(g.loc[q, "lo_percap"]))
                _hi.append(float(g.loc[q, "hi_percap"]))
        g["lo"], g["hi"] = _lo, _hi
        g.to_csv(f"{OUTDIR}/quadrant_cost_interval.csv",
                 index_label="quadrant", encoding="utf-8-sig")
        print("\n各風險象限餘生成本(象限平均 [95% 模擬區間,象限層級],萬元):")
        for q, row in g.iterrows():
            print(f"  {q}: {row['mean']/1e4:.1f} 萬 [{row['lo']/1e4:.1f}, {row['hi']/1e4:.1f}] (n={int(row['n'])})"
                  f"  〔舊口徑(個人區間取平均,已知誇大): {row['lo_percap']/1e4:.1f}–{row['hi_percap']/1e4:.1f}〕")
        plt.figure(figsize=(8.2, 4.6))
        x = np.arange(len(g))
        colors = ["#dc2626", "#f59e0b", "#3b82f6", "#16a34a"][:len(g)]
        yerr = np.vstack([(g["mean"] - g["lo"]) / 1e4, (g["hi"] - g["mean"]) / 1e4])
        # 誤差線右移 0.22、長條收窄為 0.62,避免誤差線貫穿數值標籤
        plt.bar(x, g["mean"] / 1e4, color=colors, alpha=0.85, width=0.62)
        plt.errorbar(x + 0.22, g["mean"] / 1e4, yerr=yerr, fmt="none",
                     ecolor="#333", capsize=5, lw=1.2)
        for xi, m in zip(x, g["mean"]):
            plt.text(xi - 0.06, m / 1e4 + 1.5, f"{m/1e4:.1f} 萬", ha="center",
                     va="bottom", fontsize=10, fontweight="bold")
        plt.margins(y=0.16)
        plt.xticks(x, g.index, rotation=12, fontsize=9)
        plt.ylabel(f"預估餘生醫療成本(萬元,{money_label()})")
        plt.title(f"各風險象限餘生成本({money_label()};時界:{horizon_label()})\n"
                  f"蒙地卡羅平均 + 95% 模擬區間〔象限層級:先加總再取百分位〕"
                  f"(成本源:陳興寶 2003)", fontsize=10)
        plt.tight_layout(); plt.savefig(f"{OUTDIR}/mc_cost_by_quadrant.png", dpi=150)
        if _NB: plt.show()
        plt.close()

    # ---- 圖 2:每位病患終身成本(依平均排序,含 95% 區間)----
    # [修正 2 配套] 此圖與勞倫茲曲線、母體總計一致,皆以去重後的病患為單位;
    #               全部 109 筆的個人估計仍完整輸出於 CSV。
    s = out[out["counted_in_population"]].sort_values("lifetime_mean").reset_index(drop=True)
    plt.figure(figsize=(9, 4.4))
    xx = np.arange(len(s))
    plt.fill_between(xx, s.lifetime_lo / 1e4, s.lifetime_hi / 1e4,
                     color="#c4b5fd", alpha=0.6, label="95% 區間")
    plt.plot(xx, s.lifetime_mean / 1e4, color="#6d28d9", lw=1.5, label="平均")
    plt.xlabel(f"病患(n={len(s)},依餘生成本平均排序;重複回診僅取首筆)")
    plt.ylabel(f"餘生醫療成本(萬元,{money_label()})")
    plt.title(f"每位病患餘生成本({money_label()};時界:{horizon_label()})\n"
              f"蒙地卡羅平均與 95% 模擬區間(N={N_SIM}/人)")
    plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/mc_cost_per_patient.png", dpi=150)
    if _NB: plt.show()
    plt.close()

    # ---- 圖 3:母體總終身成本的模擬分布 ----
    plt.figure(figsize=(7, 4))
    plt.hist(all_pop / 1e6, bins=30, color="#0d9488", edgecolor="white")
    for pct, lab, col in [(2.5, "2.5%", "#dc2626"), (50, "中位數", "#111"), (97.5, "97.5%", "#dc2626")]:
        v = np.percentile(all_pop, pct)
        plt.axvline(v / 1e6, color=col, ls="--", lw=1.1)
    plt.xlabel(f"母體餘生總醫療成本(百萬元,{money_label()})"); plt.ylabel("模擬次數")
    plt.title(f"母體餘生總成本的蒙地卡羅分布({money_label()};N={N_SIM} 次)\n"
              f"時界:{horizon_label()};含全體共用之母體參數不確定性")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/mc_population_total.png", dpi=150)
    if _NB: plt.show()
    plt.close()

    # ---- 圖 4:成本集中度勞倫茲曲線 + Gini 係數 ----
    # [修正 2] 集中度以「病患」為單位,不重複計入回診紀錄
    cc = np.sort(out.loc[out["counted_in_population"], "lifetime_mean"].values)
    n = len(cc)
    cum_p = np.arange(1, n + 1) / n
    cum_c = np.cumsum(cc) / cc.sum()
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz   # numpy 2.x/1.x 皆可
    # 〔稽核修正 CODE-26〕勞倫茲曲線須自原點 (0,0) 起積分;原式 _trapz(cum_c, cum_p) 自 (1/n, ·)
    #   起算,漏掉第一個梯形(面積 = cum_c[0]/(2n))。補上原點後 Gini 約減少 1.2×10⁻⁵
    #   (以 patient_lifetime_cost_mc.csv 之整數元計:0.402675 → 0.402663),三位小數(0.403)
    #   與圖上兩位小數(0.40)皆不變。make_diagnostics.py、make_figures_v2.py 同步修正。
    gini = 1 - 2 * _trapz(np.r_[0.0, cum_c], np.r_[0.0, cum_p])
    top10 = cc[::-1][:max(1, round(n * 0.10))].sum() / cc.sum() * 100
    top20 = cc[::-1][:max(1, round(n * 0.20))].sum() / cc.sum() * 100
    plt.figure(figsize=(6.2, 5.6))
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1.2, label="完全平均線")
    plt.plot(np.r_[0, cum_p], np.r_[0, cum_c], color="#6d28d9", lw=2.2,
             label=f"實際分布(Gini={gini:.2f})")
    plt.fill_between(np.r_[0, cum_p], np.r_[0, cum_c], np.r_[0, cum_p],
                     color="#c4b5fd", alpha=0.45)
    for k, share, col in [(10, top10, "#dc2626"), (20, top20, "#ea580c")]:
        x = 1 - k / 100
        y = np.interp(x, cum_p, cum_c)
        plt.plot([x, x], [0, y], ":", color=col, lw=1.2)
        plt.plot([x, 1], [y, y], ":", color=col, lw=1.2)
        plt.annotate(f"最貴前{k}%\n占 {share:.0f}%", xy=(x, y), xytext=(x - 0.34, y + 0.05),
                     fontsize=9, color=col, arrowprops=dict(arrowstyle="->", color=col, lw=1))
    plt.xlabel("累積病患比例(由成本低到高)"); plt.ylabel("累積餘生成本比例")
    plt.title(f"餘生醫療成本集中度:勞倫茲曲線(n={n} 位病患)")  # [修正 11] 原寫死 100
    plt.xlim(0, 1); plt.ylim(0, 1); plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/mc_lorenz.png", dpi=150)
    if _NB: plt.show()
    plt.close()
    print(f"[集中度] Gini={gini:.3f};最貴前 10% 占 {top10:.1f}%、前 20% 占 {top20:.1f}% → mc_lorenz.png")

    # ---- 敏感度分析:類別權重對機率校準與成本的影響 ----
    # [修正 C] 經實測,main() 的模組層級 rng 與 simulate_costs() 自建的
    #          default_rng(SEED) 抽樣序列完全一致(原版敏感度表之 cost_mean 與
    #          「全部 109 筆平均」差 0.00%),故不共用 Generator——共用反而會因
    #          main() 已推進序列而引入約 0.15% 的落差。真正的問題是「每人平均」
    #          一詞在兩處指涉不同分母,已改由欄位命名區分(見下)。
    sensitivity_class_weight(d, targets, pid, age, is_first, gender=gender)

    print(f"\n完成。輸出於 ./{OUTDIR}/  (patient_lifetime_cost_mc.csv、mc_*.png、"
          f"cost_sensitivity_class_weight.csv)")
    print(f"提醒:金額為 {TARGET_YEAR} 年人民幣(上海醫療保健類 CPI 逐年連乘校正)。")
    print("     LIFE_EXPECTANCY 已改用《中國人身保險業經驗生命表(2025)》CL2 官方")
    print("     逐歲死亡率(0~105歲、分性別)標準生命表法算出,非模型校準或占位估計。")


if __name__ == "__main__":
    main()