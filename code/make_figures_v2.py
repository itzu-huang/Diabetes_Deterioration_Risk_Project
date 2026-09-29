# -*- coding: utf-8 -*-
"""
make_figures_v2.py —— 依 v2 結果重新產生海報與簡報所需的圖
================================================================
產出至 figures_v2/。分三類:
  (a) 直接取自程式輸出的圖(SHAP、ROC、四象限、LSTM 單人圖)→ 複製並另存高解析版
  (b) 自製呈現圖(cost_bar、lorenz、lstm_scatter、groupkfold、markov_converge)
      → 依 v2 的 CSV 重繪,盡量沿用原簡報的視覺風格
  (c) 〔稽核修正 B-29〕簡報第 10、12、16、20 張原用之 5 張圖(ppt_assets/sigmoid、markov_heat、
      life_interp、gamma_cost、or_forest)原本沒有任何產生程式;現補於本檔,輸出為
      slide_sigmoid.png、slide_markov_heat.png、slide_life_interp.png、slide_gamma_cost.png、
      slide_or_forest.png(數據讀自 markov_3state.csv、life_table_ex.csv、lr_oddsratio_*.csv,
      以及 lifetime_cost_montecarlo.py 之 COST_CV)。

數字之來源(〔稽核修正 B-15〕原寫「所有數字一律讀自 output*/ 的 CSV,不寫死」並不成立;
  原 L157、L203、L264、L293、L296 有寫死之數字與結論句。現況如實列出如下):
  ・讀自 CSV:所有畫在圖上之資料值、成本、Gini、前 20% 占比、勝率、AUC 與區間、RMSE、
    轉移矩陣、生命表餘命、勝算比;重複次數(model_results.csv 之 N_splits ÷ 5);
    CGM 時步(markov_convergence_3state.csv)。
  ・讀自程式常數(以 ast 解析原始碼,不執行該程式):COST_CV、TARGET_YEAR
    (lifetime_cost_montecarlo.py);SHORT_HORIZON、HORIZONS(deterioration_risk.py)。
  ・仍為寫死者(屬呈現設定或歷史紀錄,非研究結果):
      - slide_cv_design 之示意參數(20 位病患、5 折、畫出 3 次重複)與色塊;
      - slide_markov_converge 之標註時界(1 小時、4 小時、24 小時)與「原設定」字樣;
      - slide_lorenz 之「前 20%」切點(0.8);
      - _figures_manifest.csv 中 slide_lorenz 之歷史對照值(修正前 0.400/48.1%、v1 0.426/49.2%);
      - slide_life_interp 之錨點年齡(0、20、30…105 歲,僅決定標記位置)與註解年齡 57 歲;
      - slide_or_forest 所列之 5 組(特徵→目標)配對(沿用原簡報之選擇;非跨目標之「前幾大」,
        見稽核 A-30),勝算比數值讀自 CSV;
      - slide_gamma_cost 之橫軸範圍(1−2.33×CV 至 1+3.33×CV,CV=0.30 時即原圖之 0.3–2.0);
      - slide_sigmoid 為數學示意(無資料)。
  ・原 slide_lstm_horizons 與 slide_auc_intervals 之標題為寫死之結論句,現改為依 CSV 判斷後
    產生描述句(兩圖皆未嵌入任何交付物,見稽核 C-07)。

〔稽核修正 B-16〕(a) 類之來源圖若不存在:刪除 figures_v2/ 內之同名舊副本(原本會沿用,
  互檢報告第 16 列所記「兩張 figures_v2 停在舊版」即此類事故),其餘圖照常產生,最後列出
  缺漏清單並以結束碼 1 結束;quadrant_cost_interval.csv 不存在時亦報錯(原本退回舊口徑之誤差棒)。
字型依作業系統自動選用:Windows 為微軟正黑體(Microsoft JhengHei);Linux 為思源黑體
  (Noto Sans CJK TC);故不同作業系統產生之圖檔位元組不同,屬正常。
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
import ast
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures_v2")
os.makedirs(OUT, exist_ok=True)

_av = {f.name for f in fm.fontManager.ttflist}
for _f in ["Microsoft JhengHei", "Noto Sans CJK TC", "WenQuanYi Zen Hei", "SimHei"]:
    if _f in _av:
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False

TEAL, RED, AMBER, BLUE, GREEN = "#0d9488", "#dc2626", "#f59e0b", "#3b82f6", "#16a34a"
QC = [RED, AMBER, BLUE, GREEN]
DPI_POSTER = 300      # 80x100 cm 海報用
DPI_SLIDE = 200

manifest = []
MISSING = []          # 〔稽核修正 B-16〕缺漏之來源,最後統一報錯


def save(fig, name, dpi, note):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest.append({"檔案": name, "dpi": dpi, "說明": note})
    print(f"  [fig] {name}  ({dpi} dpi)  {note}")


def _const(script, name):
    """〔稽核修正 B-15〕自分析程式原始碼讀取模組層級常數(以 ast 解析,不執行該程式,
    故無任何副作用)。找不到即報錯,不以預設值充數。"""
    src = open(os.path.join(HERE, script), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{script} 中找不到常數 {name}")


def _drop_stale(name):
    """〔稽核修正 B-16〕刪除 figures_v2/ 內之同名舊檔,避免被誤當本次產物。"""
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        os.remove(p)
        print(f"  [!] 已刪除舊副本 figures_v2/{name}(本次無法重新產生)")


# 〔稽核修正 B-15〕重複次數讀自 model_results.csv(N_splits = 重複次數 × 5 折),不再寫死 100
MR = pd.read_csv(os.path.join(HERE, "output/model_results.csv"))
_ns = sorted(set(MR.N_splits.astype(int)))
if len(_ns) != 1 or _ns[0] % 5:
    sys.exit(f"[錯誤] model_results.csv 之 N_splits 不一致或非 5 之倍數:{_ns}")
N_REP = _ns[0] // 5

# ---------------------------------------------------------------- #
# (a) 直接沿用程式輸出的圖
# ---------------------------------------------------------------- #
print("(a) 程式輸出圖")
DIRECT = [
    ("output/SHAP_Any_Complication.png", "poster_fig441_SHAP.png",
     "海報圖1 / 簡報 SHAP —— 取代 poster/fig441.png"),
    ("output_risk/risk_map_2d.png", "poster_fig481_risk_map.png",
     "海報圖2 / 簡報四象限 —— 取代 poster/fig481.png(Y 軸已改為 30 分鐘)"),
    ("output/ROC_Any_Complication.png", "slide_ROC_Any.png",
     "簡報 ROC —— 取代 app_data/output/ROC_Any_Complication.png"),
    ("cgm_output/lstm_forecast_2000.png", "slide_lstm_forecast_2000.png",
     "簡報 LSTM 單人圖 —— 取代 app_data/cgm_output/lstm_forecast_2000.png"
     "(〔稽核修正 C-02〕與 lstm_vs_baseline_all.csv 同一模型)"),
]
for src, dst, note in DIRECT:
    s = os.path.join(HERE, src)
    if os.path.exists(s):
        shutil.copy2(s, os.path.join(OUT, dst))
        manifest.append({"檔案": dst, "dpi": "原檔 150", "說明": note})
        print(f"  [fig] {dst}  (複製自 {src})")
    else:
        # 〔稽核修正 B-16〕原本只印「[!] 缺」後繼續,figures_v2 內之舊副本因而被沿用
        print(f"  [!] 缺 {src}")
        _drop_stale(dst)
        MISSING.append(src)

# ---------------------------------------------------------------- #
# (b) 自製呈現圖
# ---------------------------------------------------------------- #
print("(b) 自製呈現圖(依 v2 CSV 重繪)")
cost = pd.read_csv(os.path.join(HERE, "output_cost_mc/patient_lifetime_cost_mc.csv"))
first = cost[cost.counted_in_population]
order = ["A 立即介入(長高+短高)", "B 慢性追蹤(長高+短低)",
         "C 血糖波動注意(長低+短高)", "D 常規追蹤(長低+短低)"]
# [修正 15 配套] 象限 95% 區間改用象限層級百分位(quadrant_cost_interval.csv)。
# 原式 lo/hi = 各病患自身 2.5/97.5 百分位再取平均,長條高度估的是象限平均、
# 誤差棒估的卻是個人成本之離散度,不是同一個量,寬度被誇大 2.1–3.3 倍。
# 〔稽核修正 B-15〕原寫「2.2–3.4 倍」係拿舊年金+舊口徑之區間相除,混入年金修正之效果;
#   同一(修正後)年金下,(hi_percap−lo_percap)/(hi−lo) = 2.20/2.10/3.07/3.25(稽核 A-18)。
g = first.groupby("quadrant").agg(n=("record", "size"), mean=("lifetime_mean", "mean"),
                                  lo=("lifetime_lo", "mean"), hi=("lifetime_hi", "mean")
                                  ).reindex(order).dropna()
_qci = os.path.join(HERE, "output_cost_mc/quadrant_cost_interval.csv")
_cost_ok = os.path.exists(_qci)
if _cost_ok:
    _Q = pd.read_csv(_qci).set_index("quadrant")
    g["lo"] = [_Q.loc[q, "lo"] for q in g.index]
    g["hi"] = [_Q.loc[q, "hi"] for q in g.index]
else:
    # 〔稽核修正 B-16〕原本印警告後改用舊口徑(已知誇大)之誤差棒照畫;改為不產生該圖並報錯
    print("[錯誤] 找不到 output_cost_mc/quadrant_cost_interval.csv(請先執行 lifetime_cost_montecarlo.py);"
          "slide_cost_bar.png 不產生。")
    _drop_stale("slide_cost_bar.png")
    MISSING.append("output_cost_mc/quadrant_cost_interval.csv")
_TY = _const("lifetime_cost_montecarlo.py", "TARGET_YEAR")   # 〔稽核修正 B-15〕原寫死 2024

# --- cost_bar:各象限餘生成本 ---
if _cost_ok:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(g))
    ax.bar(x, g["mean"] / 1e4, color=QC[:len(g)], alpha=.9, width=.62)
    yerr = np.vstack([(g["mean"] - g["lo"]) / 1e4, (g["hi"] - g["mean"]) / 1e4])
    ax.errorbar(x, g["mean"] / 1e4, yerr=yerr, fmt="none", ecolor="#333", capsize=6, lw=1.2)
    for xi, (m, n) in enumerate(zip(g["mean"], g["n"])):
        ax.text(xi, m / 1e4 + 2, f"{m/1e4:.1f} 萬", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
        ax.text(xi, 1.5, f"n={int(n)}", ha="center", fontsize=9, color="white",
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([q.split("(")[0] for q in g.index], fontsize=10)
    ax.set_ylabel(f"餘生成本(萬元,{_TY} 人民幣)")
    ax.margins(y=.22)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "slide_cost_bar.png", DPI_SLIDE,
         "取代 ppt_assets/cost_bar.png —— 各象限餘生成本"
         "(平均 + 95% 模擬區間〔象限層級:先加總再取百分位〕)")

# --- lorenz:成本集中度 ---
cc = np.sort(first["lifetime_mean"].values)
n = len(cc)
cum_p = np.arange(1, n + 1) / n
cum_c = np.cumsum(cc) / cc.sum()
# 〔稽核修正 CODE-26〕勞倫茲曲線自原點 (0,0) 起積分(原式漏第一個梯形,差約 1.2×10⁻⁵;三位小數不變)
gini = 1 - 2 * np.trapezoid(np.r_[0.0, cum_c], np.r_[0.0, cum_p])
top20 = cc[::-1][:round(n * .2)].sum() / cc.sum() * 100
fig, ax = plt.subplots(figsize=(6.2, 5.4))
ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1.4, label="完全均等")
ax.plot(np.r_[0, cum_p], np.r_[0, cum_c], color=RED, lw=2.4, label="實際分布")
ax.fill_between(np.r_[0, cum_p], np.r_[0, cum_c], np.r_[0, cum_p],
                color=RED, alpha=.13)
xx = .8
yy = np.interp(xx, cum_p, cum_c)
ax.plot([xx, xx], [0, yy], ":", color="#334155", lw=1.2)
ax.annotate(f"前 20% 佔 {top20:.1f}%", xy=(xx, yy), xytext=(.30, .60), fontsize=11,
            color="#334155", arrowprops=dict(arrowstyle="->", color="#334155", lw=1.1))
ax.text(.52, .16, f"吉尼係數 {gini:.3f}", fontsize=13, fontweight="bold", color=TEAL)
ax.set_xlabel("病患累積比例(成本由低至高)")
ax.set_ylabel("累積成本比例")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(loc="upper left", fontsize=10, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "slide_lorenz.png", DPI_SLIDE,
     f"取代 ppt_assets/lorenz_pres.png —— Gini {gini:.3f}、前20% {top20:.1f}%"
     f"(修正前之交付物為 0.400 / 48.1%;更早之 v1 為 0.426 / 49.2%)")

# --- lstm_scatter:逐人 vs 合併訓練 ---
A = pd.read_csv(os.path.join(HERE, "cgm_output/lstm_vs_baseline_all.csv"))[
    ["record", "skill_pct"]].rename(columns={"skill_pct": "per_patient"})
P = pd.read_csv(os.path.join(HERE, "cgm_output/lstm_pooled_vs_baseline.csv"))[
    ["record", "skill_pct"]].rename(columns={"skill_pct": "pooled"})
M = A.merge(P, on="record")
above = (M.pooled > M.per_patient).mean() * 100
neg = (M.per_patient < 0).sum()
fig, ax = plt.subplots(figsize=(6.4, 5.2))
lim = [min(M.per_patient.min(), M.pooled.min()) - 6,
       max(M.per_patient.max(), M.pooled.max()) + 6]
ax.plot(lim, lim, "--", color="#94a3b8", lw=1.2)
ax.axhline(0, color="#cbd5e1", lw=1)
ax.axvline(0, color="#cbd5e1", lw=1)
ax.scatter(M.per_patient, M.pooled, s=30, alpha=.75, color=TEAL, edgecolor="white", lw=.6)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("逐人專屬之技巧分數(%)")
ax.set_ylabel("合併訓練之技巧分數(%)")
ax.text(.03, .95, f"{above:.0f}% 位於對角線上方\n(合併訓練較佳)", transform=ax.transAxes,
        va="top", fontsize=11, color=TEAL, fontweight="bold")
ax.text(.03, .06, f"另有 {neg} 筆 <0%\n(逐人版劣於基準)", transform=ax.transAxes,   # 〔稽核修正 A-24〕單位為監測紀錄(筆)
        fontsize=9.5, color="#64748b")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "slide_lstm_scatter.png", DPI_SLIDE,
     f"取代 ppt_assets/lstm_scatter.png —— 合併訓練勝出 {above:.0f}%、逐人版 {neg} 位為負")

# --- groupkfold → 新的 CV 示意 ---
rng = np.random.RandomState(42)
fig, ax = plt.subplots(figsize=(8.4, 4.2))
n_pat, n_fold, n_rep = 20, 5, 3
for r in range(n_rep):
    perm = rng.permutation(n_pat)
    assign = np.empty(n_pat, int)
    for k, j in enumerate(perm):
        assign[j] = k % n_fold
    for p in range(n_pat):
        for f in range(n_fold):
            test = (assign[p] == f)
            ax.add_patch(plt.Rectangle(
                (p, r * (n_fold + 1) + f), .92, .92,
                color=TEAL if test else "#e2e8f0"))
    ax.text(-1.4, r * (n_fold + 1) + n_fold / 2 - .5, f"重複 {r+1}",
            ha="right", va="center", fontsize=10, fontweight="bold")
# 〔稽核修正 B-15〕重複次數讀自 model_results.csv(原寫死 100)
ax.text(n_pat / 2, n_rep * (n_fold + 1) + .4, f"…共 {N_REP} 次重複(random_state = SEED + r)",
        ha="center", fontsize=10, color="#475569")
ax.set_xlim(-6, n_pat + .5)
ax.set_ylim(-1.2, n_rep * (n_fold + 1) + 1.6)
ax.set_xlabel("每一欄 = 一位病患(同一病患的多次回診永遠同組)", fontsize=10)
ax.set_yticks([])
ax.set_xticks([])
ax.set_frame_on(False)
ax.add_patch(plt.Rectangle((-5.6, -1.0), .8, .8, color=TEAL))
ax.text(-4.5, -.6, "該折之驗證集", fontsize=9.5, va="center")
ax.add_patch(plt.Rectangle((-5.6 + 5.2, -1.0), .8, .8, color="#e2e8f0"))
ax.text(-4.5 + 5.2, -.6, "訓練集", fontsize=9.5, va="center")
ax.set_title(f"StratifiedGroupKFold(5 折,shuffle=True)× {N_REP} 次重複",
             fontsize=12, fontweight="bold")
save(fig, "slide_cv_design.png", DPI_SLIDE,
     "取代 ppt_assets/groupkfold.png —— 原圖示意 GroupKFold(未設 shuffle,折分配不可重現)")
# 〔稽核修正 B-15〕原說明寫「GroupKFold(不接受 random_state)」,scikit-learn 1.6 起已不成立

# --- markov_converge:加標 30 分鐘位置 ---
C3 = pd.read_csv(os.path.join(HERE, "cgm_output/markov_convergence_3state.csv"))
C3 = C3[C3.k_steps.astype(str) != "stationary"].copy()
C3["k"] = C3.k_steps.astype(int)
# 〔稽核修正 B-15〕每步分鐘數讀自 CSV(k=1 之 hours × 60),原寫死 15
STEP_MIN = float(C3.loc[C3.k == 1, "hours"].astype(float).iloc[0]) * 60
C3["hours"] = C3["k"] * STEP_MIN / 60
C3["out"] = C3["Low(<70)"] + C3["High(>180)"]
stat = pd.read_csv(os.path.join(HERE, "cgm_output/markov_stationary_3state.csv"))
pi_out = float(stat["Low(<70)"][0]) + float(stat["High(>180)"][0])
# 〔稽核修正 B-15〕「本研究採用」之時界讀自 deterioration_risk.py 之 SHORT_HORIZON(原寫死 k=2)
_SH = _const("deterioration_risk.py", "SHORT_HORIZON")
_K_ADOPT = _const("deterioration_risk.py", "HORIZONS")[_SH]
_lab_adopt = f"{int(_K_ADOPT * STEP_MIN)} 分鐘\n(本研究採用)"
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(C3.hours, C3.out, "-o", color=TEAL, lw=2.2, ms=5)
ax.axhline(pi_out, ls="--", color=RED, lw=1.4)
ax.text(C3.hours.max(), pi_out + .5, f"穩態 {pi_out:.1f}%", ha="right",
        color=RED, fontsize=10.5, fontweight="bold")
for k, lab, col in [(_K_ADOPT, _lab_adopt, RED), (4, "1 小時", "#475569"),
                    (16, "4 小時", "#475569"), (96, "24 小時\n(原設定,已收斂)", "#475569")]:
    row = C3[C3.k == k]
    if len(row) == 0:
        continue
    h, v = float(row.hours.iloc[0]), float(row.out.iloc[0])
    ax.plot([h], [v], "o", ms=9, mfc="none", mec=col, mew=2)
    ax.annotate(f"{lab}\n{v:.1f}%", xy=(h, v), xytext=(h + 1.2, v - 3.4),
                fontsize=9.5, color=col,
                arrowprops=dict(arrowstyle="->", color=col, lw=1))
ax.set_xlabel("經過時間(小時)")
ax.set_ylabel("出範圍機率(%)")
ax.set_title("馬可夫鏈之收斂:時界越長,推算越趨近狀態時間占比", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "slide_markov_converge.png", DPI_SLIDE,
     "取代 ppt_assets/markov_converge.png —— 加標 30 分鐘採用點與收斂說明")

# --- 新增:LSTM 各預測時界對照(文獻可比) ---
H = pd.read_csv(os.path.join(HERE, "cgm_output/lstm_pooled_by_horizon.csv"))
fig, ax = plt.subplots(figsize=(7.4, 4.4))
x = np.arange(len(H))
w = .36
ax.bar(x - w / 2, H.baseline_rmse_mean, w, color="#94a3b8", label="持續性基準")
ax.bar(x + w / 2, H.lstm_rmse_mean, w, color=TEAL, label="合併訓練 LSTM")
for xi, (b, l, s) in enumerate(zip(H.baseline_rmse_mean, H.lstm_rmse_mean, H.skill_median)):
    ax.text(xi - w / 2, b + .4, f"{b:.2f}", ha="center", fontsize=9)
    ax.text(xi + w / 2, l + .4, f"{l:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(xi, max(b, l) + 2.0, f"改善 {s:.1f}%", ha="center", fontsize=10, color=RED)
ax.set_xticks(x)
ax.set_xticklabels([f"PH = {int(p)} 分" for p in H.PH_min])
ax.set_ylabel("RMSE (mg/dL)")
# 〔稽核修正 B-15〕原標題為寫死之結論句;改為依 CSV 判斷趨勢後產生
_Hs = H.sort_values("PH_min")
_b_up = bool(np.all(np.diff(_Hs.baseline_rmse_mean.values) > 0))
_s_dn = bool(np.all(np.diff(_Hs.skill_median.values) < 0))
if _b_up and _s_dn:
    _ttl = "預測時界越長,持續性基準越弱,但 LSTM 之相對改善也縮小"
else:
    _ttl = "合併訓練 LSTM 與持續性基準於各預測時界之 RMSE(逐紀錄平均)"
ax.set_title(_ttl, fontsize=11.5)
ax.legend(fontsize=10, frameon=False)
ax.margins(y=.18)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "slide_lstm_horizons.png", DPI_SLIDE,
     "新增 —— LSTM 於 PH=15/30/60 分之表現(文獻慣用時界,可對照)")

# --- 新增:AUC 區間圖(取代單點數字) ---
fig, ax = plt.subplots(figsize=(7.6, 4.2))
targets = ["Any_Complication", "Microvascular", "Macrovascular"]
labels = ["任一併發症", "小血管", "大血管"]
ypos = 0
yt, yl = [], []
for t, lab in zip(targets, labels):
    for mdl, col, nm in [("Logistic Regression (baseline)", BLUE, "LR"),
                         (MR[MR.Target == t].Model.iloc[1], RED, "XGBoost")]:
        r = MR[(MR.Target == t) & (MR.Model == mdl)].iloc[0]
        m, lo, hi = r.AUC_rep_median, r.AUC_rep_lo95, r.AUC_rep_hi95
        ax.plot([lo, hi], [ypos, ypos], color=col, lw=3, solid_capstyle="round", alpha=.45)
        ax.plot([m], [ypos], "o", color=col, ms=9)
        ax.text(hi + .004, ypos, f"{m:.3f} [{lo:.3f}, {hi:.3f}]", va="center", fontsize=9.5)
        yt.append(ypos)
        yl.append(f"{lab} · {nm}")
        ypos -= 1
    ypos -= .5
ax.axvline(.5, color="#cbd5e1", lw=1)
ax.set_yticks(yt)
ax.set_yticklabels(yl, fontsize=10)
# 〔稽核修正 B-15〕重複次數讀自 CSV(原寫死 100);原標題「小血管目標之兩模型區間高度重疊,
#   差異未達可分辨程度」為寫死之結論句,且此區間只反映折分配之變異、非抽樣信賴區間(稽核 A-12),
#   改為描述句。
ax.set_xlabel(f"AUC({N_REP} 次重複之中位數與 2.5–97.5 百分位)")
ax.set_xlim(.6, 1.02)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_title(f"模型效能:{N_REP} 次隨機折分配之 AUC(區間反映折分配之變異,非信賴區間)",
             fontsize=11.5)
save(fig, "slide_auc_intervals.png", DPI_SLIDE,
     "新增 —— 取代單點 AUC 表,呈現折分配不確定性")

# ---------------------------------------------------------------- #
# (c) 〔稽核修正 B-29〕簡報用而原無產生程式之 5 張圖
# ---------------------------------------------------------------- #
print("(c) 簡報用圖(原 ppt_assets,補產生程式)")
_INK, _GREY = "#1e3238", "#808080"
_TEAL2, _TEAL2D = "#0b7c85", "#075e66"
_RED2, _GREEN2 = "#d64550", "#3e9b6b"

# --- 簡報第 10 張:sigmoid(數學示意,無資料)---
z = np.linspace(-6, 6, 400)
fig, ax = plt.subplots(figsize=(4.9, 3.06))
ax.plot(z, 1 / (1 + np.exp(-z)), color=_TEAL2, lw=3.2)
ax.axhline(.5, ls="--", color=_GREY, lw=1.1)
ax.axvline(0, ls="--", color=_GREY, lw=1.1)
ax.text(-5.9, .86, r"$p=\sigma(z)=\dfrac{1}{1+e^{-z}}$", fontsize=14, color=_TEAL2D)
ax.text(5.9, .1, r"$z=\beta_0+\sum_i \beta_i z_i$", fontsize=13, color=_INK, ha="right")
ax.set_xlim(-6.6, 6.6)
ax.set_ylim(0, 1)
ax.set_xticks(range(-6, 7, 2))
ax.set_xlabel("線性分數 z", fontsize=11)
ax.set_ylabel("併發症機率 p", fontsize=11)
save(fig, "slide_sigmoid.png", DPI_SLIDE,
     "取代 ppt_assets/sigmoid.png(簡報第 10 張)—— 邏輯斯函數示意(數學式,無資料)")

# --- 簡報第 12 張:三態轉移矩陣熱圖(讀 markov_3state.csv)---
M3 = pd.read_csv(os.path.join(HERE, "cgm_output/markov_3state.csv"), index_col=0)
_zh = {"Low(<70)": "低血糖", "InRange(70-180)": "範圍內", "High(>180)": "高血糖"}
Pm = M3.values.astype(float)
fig, ax = plt.subplots(figsize=(4.9, 3.76))
ax.imshow(Pm, cmap="Blues", vmin=0, vmax=1)
for i in range(Pm.shape[0]):
    for j in range(Pm.shape[1]):
        ax.text(j, i, f"{Pm[i, j]:.3f}", ha="center", va="center", fontsize=13,
                fontweight="bold", color="white" if Pm[i, j] > .5 else _INK)
ax.set_xticks(range(Pm.shape[1]))
ax.set_xticklabels([_zh.get(c, c) for c in M3.columns], fontsize=11)
ax.set_yticks(range(Pm.shape[0]))
ax.set_yticklabels([_zh.get(c, c) for c in M3.index], fontsize=11)
ax.set_xlabel(f"下一步（{int(STEP_MIN)} 分鐘後）", fontsize=12)
ax.set_ylabel("目前狀態", fontsize=12)
save(fig, "slide_markov_heat.png", DPI_SLIDE,
     "取代 ppt_assets/markov_heat.png(簡報第 12 張)—— 三態轉移矩陣(讀自 cgm_output/markov_3state.csv)")

# --- 簡報第 16 張(左):生命表平均餘命(讀 life_table_ex.csv)---
LT = pd.read_csv(os.path.join(HERE, "output_cost_mc/life_table_ex.csv"))
_ANCH = [0, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]   # 標記位置(呈現用)
_mk = [int(i) for i in LT.index[LT.age.isin(_ANCH)]]
_EX_AGE = 57                                                                  # 註解之示範年齡
_ex57 = float(LT.loc[LT.age == _EX_AGE, "ex_male"].iloc[0])
fig, ax = plt.subplots(figsize=(8.75, 5.42))
ax.plot(LT.age, LT.ex_male, "-o", color="#0e7c86", lw=2.4, ms=6.5, markevery=_mk, label="男性")
ax.plot(LT.age, LT.ex_female, "-o", color="#d6455b", lw=2.4, ms=6.5, markevery=_mk, label="女性")
ax.plot([_EX_AGE], [_ex57], "o", color="#123b4a", ms=9, zorder=5)
ax.annotate(f"{_EX_AGE}歲男性→{_ex57:.1f}年", xy=(_EX_AGE, _ex57), xytext=(_EX_AGE + 6, _ex57 + 16),
            fontsize=11, color="#123b4a",
            arrowprops=dict(arrowstyle="->", color="#123b4a", lw=1.1))
ax.set_xlim(-3, 108)
ax.set_ylim(0, None)
ax.set_xlabel("年齡", fontsize=12)
ax.set_ylabel("平均餘命（年）", fontsize=12)
ax.set_title("中國人身保險業經驗生命表(2025) 非養老類業務一表", fontsize=13)
ax.legend(fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "slide_life_interp.png", DPI_SLIDE,
     "取代 ppt_assets/life_interp.png(簡報第 16 張左)—— 0–105 歲逐歲 e(x)"
     "(讀自 output_cost_mc/life_table_ex.csv;圓點標記 16 個錨點年齡)")

# --- 簡報第 16 張(右):Gamma 成本分布(CV 讀自 lifetime_cost_montecarlo.py)---
from scipy.stats import gamma as _gamma
_CV = float(_const("lifetime_cost_montecarlo.py", "COST_CV"))
xg = np.linspace(max(.01, 1 - 2.33 * _CV), 1 + 3.33 * _CV, 400)
yg = _gamma.pdf(xg, a=1 / _CV ** 2, scale=_CV ** 2)       # 平均 1、變異係數 CV
fig, ax = plt.subplots(figsize=(4.4, 2.7))
ax.plot(xg, yg, color=_RED2, lw=3.2)
ax.fill_between(xg, 0, yg, color=_RED2, alpha=.15, lw=0)
ax.axvline(1.0, ls="--", color=_GREY, lw=1.2)
ax.set_yticks([])
ax.set_ylim(0, None)
ax.set_xlabel("成本 ÷ 平均成本", fontsize=12)
ax.set_title(f"Gamma 分布（變異係數 {_CV:.2f}）", fontsize=13)
save(fig, "slide_gamma_cost.png", DPI_SLIDE,
     f"取代 ppt_assets/gamma_cost.png(簡報第 16 張右)—— 年成本 Gamma 抽樣分布(COST_CV={_CV:g},"
     f"讀自 lifetime_cost_montecarlo.py)")

# --- 簡報第 20 張:勝算比(讀 lr_oddsratio_*.csv)---
# 所列 5 組(特徵→目標)沿用原簡報之選擇;非跨目標之「前幾大」(稽核 A-30:小血管之抽菸 2.77
# 大於大血管之年齡 2.71),數值一律讀自 CSV。勝算比為全資料配適之 LR(L2、C=1)係數之指數。
_PAIRS = [("Duration", "Microvascular", "罹病年數"), ("Ins_f", "Any_Complication", "空腹胰島素"),
          ("Age", "Macrovascular", "年齡"), ("HDL", "Macrovascular", "高密度脂蛋白"),
          ("Cpep_f", "Microvascular", "空腹 C-胜肽")]
_TZH = {"Microvascular": "小血管", "Any_Complication": "任一", "Macrovascular": "大血管"}
_ors = []
for feat, tgt, zh in _PAIRS:
    _t = pd.read_csv(os.path.join(HERE, f"output/lr_oddsratio_{tgt}.csv")).set_index("Feature")
    _ors.append((f"{zh} →{_TZH[tgt]}", float(_t.loc[feat, "OddsRatio"])))
fig, ax = plt.subplots(figsize=(5.5, 3.4))
for i, (lab, v) in enumerate(_ors):
    y = len(_ors) - 1 - i
    col, lcol = (_RED2, "#ecabb0") if v > 1 else (_GREEN2, "#a8d1bc")
    ax.plot([1, v], [y, y], color=lcol, lw=4, solid_capstyle="butt", zorder=1)
    ax.plot([v], [y], "o", color=col, ms=13, zorder=3)
    ax.text(v * (1.07 if v > 1 else 1 / 1.07), y, f"{v:.2f}", va="center",
            ha="left" if v > 1 else "right", fontsize=13, fontweight="bold", color=col)
ax.axvline(1, ls="--", color="#5e747c", lw=1.6, zorder=0)
ax.set_xscale("log")
_lo = min(v for _, v in _ors) / 1.6
_hi = max(v for _, v in _ors) * 1.35
ax.set_xlim(_lo, _hi)
ax.set_xticks([.5, 1, 2, 3])
ax.set_xticklabels(["0.5", "1", "2", "3"])
ax.set_yticks(range(len(_ors)))
ax.set_yticklabels([lab for lab, _ in _ors][::-1], fontsize=11)
ax.set_ylim(-.5, len(_ors) - .5)
ax.set_xlabel("勝算比（每＋1 標準差；對數尺度）", fontsize=12)
# 〔稽核修正 A-30〕原標籤「風險下降／風險上升」帶因果意味;勝算比為全資料配適之關聯,改為中性標示
ax.text(1 / 1.04, len(_ors) - .1, "勝算比 <1 ←", ha="right", va="bottom", fontsize=12,
        color=_GREEN2, transform=ax.transData)
ax.text(1 * 1.04, len(_ors) - .1, "→ 勝算比 >1", ha="left", va="bottom", fontsize=12,
        color=_RED2, transform=ax.transData)
save(fig, "slide_or_forest.png", DPI_SLIDE,
     "取代 ppt_assets/or_forest.png(簡報第 20 張)—— 5 組(特徵→目標)勝算比(讀自 output/lr_oddsratio_*.csv;"
     "配對沿用原簡報之選擇)")

pd.DataFrame(manifest).to_csv(os.path.join(OUT, "_figures_manifest.csv"),
                              index=False, encoding="utf-8-sig")
print(f"\n完成:{len(manifest)} 張 → figures_v2/")
if MISSING:   # 〔稽核修正 B-16〕有缺漏即以非 0 結束,並列出清單
    print("[錯誤] 以下來源不存在,對應之圖未產生(figures_v2 內之同名舊副本已刪除):")
    for m in MISSING:
        print(f"   - {m}")
    sys.exit(1)
