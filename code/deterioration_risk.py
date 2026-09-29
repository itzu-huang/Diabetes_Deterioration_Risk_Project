# -*- coding: utf-8 -*-
"""
糖尿病惡化風險綜合評估
============================================================
把分散的分析整合成「可解讀的惡化風險」,分兩個時間尺度(避免主觀加權):

  長期惡化風險(慢性併發症)= 分類模型對「任一併發症」的預測機率
      - 以 Logistic Regression + StratifiedGroupKFold(5, shuffle=True,
        random_state=SEED+r) × N_REPEATS 次重複(依病患分組)之 out-of-fold
        平均機率為分數 [修正 21:原文寫 GroupKFold,與 [修正 A] 後之實作不符]
        (每位病患的分數皆來自沒看過他的模型,較不高估)。
      - 依三分位切成 低 / 中 / 高。

  短期惡化風險(血糖失控)= 由每位病患自己的馬可夫轉移矩陣,推算「未來 k 步後
      落在高血糖(>180)或低血糖(<70)」的機率(P^k,起點為 InRange)。
      - 計算 15 分(1 步)、30 分(2 步)、60 分(4 步)、90 分(6 步)四個文獻慣用時界,
        另輸出 4 小時(16 步)與 24 小時(96 步)供收斂對照(見 HORIZONS)。
      - 以 30 分鐘(SHORT_HORIZON = "30m")之出範圍機率作為短期風險分數(短期主軸)。
  〔稽核修正 B-15〕原檔頭寫「計算 1/4/24 小時三個時界、以 24 小時(≈長期穩態)之出範圍
    機率作為短期風險分數」,為 [修正 B]/[修正 B2] 之前之舊設計,與現行實作相反;
    [修正 21] 所稱「全部改為一致」於此未完成,現已改正(程式本身未改動)。

  綜合:以兩軸中位數切成四象限,對應不同臨床處置,供高齡友善決策 App 使用。

依賴:pandas, numpy, scikit-learn, matplotlib;需與 diabetes_deterioration_pipeline.py
同目錄(重用 load_data)。CGM 讀 .xls 需 xlrd,或先轉成 .xlsx。
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
import glob
import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython(); _NB = True
except NameError:
    matplotlib.use("Agg"); _NB = False
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

# 中文字型設定:自動選用系統可用者(Windows 微軟正黑體;Linux/Mac 其他)
# 若圖上中文顯示為方框「□」,就是這裡沒選到字型——把你系統有的字型名加到最前面即可。
_available = {f.name for f in _fm.fontManager.ttflist}
for _f in ["Microsoft JhengHei", "Microsoft YaHei", "PingFang TC", "Heiti TC",
           "Noto Sans CJK TC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]:
    if _f in _available:
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False   # 讓負號正常顯示

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedGroupKFold,
                                     cross_val_predict)   # 〔稽核修正 CODE-26〕刪除未使用之 GroupKFold

from diabetes_deterioration_pipeline import load_data

SEED = 42
SUMMARY_XLSX = "Shanghai_T2DM_Summary.xlsx"
CGM_DIR = "Shanghai_T2DM"          # 放 CGM 檔的資料夾
OUTDIR = "output_risk"
STEP_MIN = 15                       # CGM 取樣間隔(分)
# [修正 B2] 時界改採 CGM 血糖預測文獻之慣用 prediction horizon(15/30/60/90 分),
#           而非原本的 1h/4h/24h。ATTD 國際共識(Battelino 2019 Diabetes Care;
#           2023 Lancet Diabetes Endocrinol 臨床試驗版)規範的是 TIR/TBR/TAR/GMI/CV
#           等「指標定義」,並未規範預測時界;預測時界的事實標準來自 CGM 預測文獻,
#           以 30 分與 60 分最為通行(低血糖預測多用 30 分)。4h/24h 保留作收斂對照。
HORIZONS = {"15m": 1, "30m": 2, "60m": 4, "90m": 6, "4h": 16, "24h": 96}
SHORT_HORIZON = "30m"   # 主要短期風險時界(文獻最通行)
os.makedirs(OUTDIR, exist_ok=True)


def _show():
    if _NB:
        plt.show()
    plt.close()


# ---------- 1. 長期風險:分類模型 out-of-fold 機率 ----------
N_REPEATS = 100   # [修正 A] 折分配重複次數


def long_term_risk():
    """[修正 A] 原本以 GroupKFold(5) 產生單一組 out-of-fold 機率。GroupKFold(未設 shuffle;
    〔稽核修正 B-15〕原寫「不接受 random_state」,scikit-learn 1.6 起已支援)之折分配由
    np.argsort(各組樣本數) 的平手順序決定(本資料 100 組中
    92 組 size=1),不可跨平台重現;實測同一位病患的 long_risk 最大可差 0.822。
    改為 StratifiedGroupKFold(shuffle=True, random_state=SEED+r) 重複 N_REPEATS 次,
    取每位病患在各次重複中的 out-of-fold 機率平均。如此:
      (1) random_state 真正生效,可跨平台重現;
      (2) 個人分數不再取決於單一次切分的運氣,穩定度大幅提高。
    同時輸出各次重複的標準差 long_risk_sd,供報告呈現個人分數的不確定性。"""
    d, med, targets, pid, _ = load_data()
    raw = pd.read_excel(SUMMARY_XLSX, sheet_name="T2DM")
    record_id = raw["Patient Number"].astype(str).values      # 與 d 同列序
    y = targets["Any_Complication"].astype(int)
    X = d.copy()                                              # 用藥洩漏特徵已排除
    lr = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=5000, class_weight="balanced",
                                              solver="liblinear", random_state=SEED))])
    P = np.empty((N_REPEATS, len(y)))
    for r in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)
        P[r] = cross_val_predict(lr, X, y, cv=cv, groups=pid,
                                 method="predict_proba")[:, 1]
    return pd.DataFrame({"record": record_id, "patient": pid.values,
                         "long_risk": P.mean(axis=0),
                         "long_risk_sd": P.std(axis=0),
                         "long_risk_lo95": np.percentile(P, 2.5, axis=0),
                         "long_risk_hi95": np.percentile(P, 97.5, axis=0),
                         "has_complication": y.values})


# ---------- 2. 短期風險:每位病患的馬可夫 P^k ----------
def load_cgm(fp):
    df = pd.read_excel(fp)
    dc = [c for c in df.columns if str(c).strip().lower().startswith("date")][0]
    gc = [c for c in df.columns if "cgm" in str(c).lower()][0]
    s = df[[dc, gc]].copy(); s.columns = ["ts", "cgm"]
    s["ts"] = pd.to_datetime(s["ts"], errors="coerce")
    s["cgm"] = pd.to_numeric(s["cgm"], errors="coerce")
    return s.dropna().sort_values("ts").reset_index(drop=True)


def st3(v):
    return 0 if v < 70 else (1 if v <= 180 else 2)   # 0=Low 1=InRange 2=High


MAX_GAP_MIN = 30   # 與 cgm_lstm_markov.py 之 markov_matrix() 採同一門檻,兩支程式口徑一致


def transition_matrix(states, ts=None, max_gap_min=MAX_GAP_MIN):
    """[修正 5] 原版僅接受 ts 參數卻從未使用,跨斷點(如感測器拔除重接)的假轉移
    未被濾除;與 cgm_lstm_markov.py 之 markov_matrix() 口徑不一致。此處補上與該
    函式相同的時間間隔過濾:間隔 >max_gap_min 分鐘的相鄰配對不計入轉移計數。
    ts=None 時(未提供時間戳)退回原行為,以維持向後相容。"""
    T = np.zeros((3, 3))
    if ts is not None:
        gap_min = pd.Series(ts).diff().dt.total_seconds().div(60).values
        for i in range(1, len(states)):
            if np.isfinite(gap_min[i]) and gap_min[i] <= max_gap_min:
                T[states[i - 1], states[i]] += 1
    else:
        for i in range(1, len(states)):
            T[states[i - 1], states[i]] += 1
    P = np.zeros((3, 3))
    for i in range(3):
        s = T[i].sum()
        P[i] = T[i] / s if s > 0 else np.eye(3)[i]   # 未出現的狀態 → 維持自身
    return P


def project_from(P, state_idx, k):
    """[新增] 給定轉移矩陣與『任意起始狀態』,推算 k 步後的狀態分布。
    供「從病患目前實際狀態」推算未來走勢使用(即時提醒情境);
    與 short_term_risk() 中固定以 InRange 為起點、用於分層之 short_risk 為互補用途,
    非取代關係——後者之所以固定起點,是刻意估計『血糖失控之穩定傾向』而非
    某一時刻的前瞻預測(見表 4-7-4 之收斂性質討論)。"""
    v = np.eye(3)[state_idx]
    return v @ np.linalg.matrix_power(P, k)


def short_term_risk():
    files = sorted(glob.glob(os.path.join(CGM_DIR, "*.xlsx")) +
                   glob.glob(os.path.join(CGM_DIR, "*.xls")))
    rows = []
    for fp in files:
        rid = os.path.splitext(os.path.basename(fp))[0]
        s = load_cgm(fp)
        g = s["cgm"].values
        if len(g) < 20:
            continue
        states = [st3(v) for v in g]
        # [修正 5] 傳入 ts 以啟用斷點過濾(見 transition_matrix 說明)
        P = transition_matrix(states, ts=s["ts"].values)
        rec = {"record": rid, "patient": rid.split("_")[0]}
        for lab, k in HORIZONS.items():
            Pk = np.linalg.matrix_power(P, k)
            rec[f"hypo_{lab}"] = Pk[1, 0] * 100    # 起點固定 InRange → Low(供分層用,見下)
            rec[f"hyper_{lab}"] = Pk[1, 2] * 100   # 起點固定 InRange → High
        # 〔稽核修正 B-15〕短期風險分數 = SHORT_HORIZON(30 分鐘,k=2)後之出範圍機率
        # (原註解「= 24h 後出範圍機率(≈ 穩態)」為 [修正 B] 前之舊文)。分層任務刻意採
        # 固定起點(InRange),詳見表 4-7-4 收斂性質討論。
        # [修正 B] 原以 24h(k=96)之出範圍機率為短期風險分數。程式自身的收斂診斷
        #          已顯示 k=96 時與穩態差距為 0.00pp,亦即該值等同狀態時間占比,
        #          不再帶有起點資訊。實測 corr(short_risk_24h, 100-TIR)=0.9995,
        #          此軸並未提供超出 TIR 的資訊。現行版本改以 30m(k=2)為主要短期軸——
        #          此時與穩態差距仍有 16.07pp,確實反映「自目前狀態出發」的短期動態。
        #          （詳見上方 [修正 B2]；本段保留以說明設計沿革。）
        #          24h 與 4h 版本一併保留供對照與敏感度分析。
        for _h in HORIZONS:
            rec[f"short_risk_{_h}"] = rec[f"hypo_{_h}"] + rec[f"hyper_{_h}"]
        rec["short_risk"] = rec[f"short_risk_{SHORT_HORIZON}"]

        # [新增] 從「病患最後一筆實際狀態」出發的推算,供即時提醒情境使用
        # (與上面固定 InRange 起點之分層用途互補,不互相取代)
        last_state = states[-1]
        for lab, k in HORIZONS.items():
            v = project_from(P, last_state, k)
            rec[f"hypo_{lab}_fromlast"] = v[0] * 100
            rec[f"hyper_{lab}_fromlast"] = v[2] * 100
        rec["last_glucose"] = float(g[-1])
        rec["last_state"] = ["Low", "InRange", "High"][last_state]

        rec["TIR"] = np.mean((g >= 70) & (g <= 180)) * 100
        rec["TBR"] = np.mean(g < 70) * 100
        rec["TAR"] = np.mean(g > 180) * 100
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------- 3. 合成與二維風險圖 ----------
def main():
    L = long_term_risk()
    S = short_term_risk()
    M = L.merge(S, on=["record", "patient"], how="inner")
    print(f"合併後樣本:{len(M)} 筆(長期 {len(L)}、短期 {len(S)})")

    # 三分位分組
    M["long_group"] = pd.qcut(M["long_risk"], 3, labels=["低", "中", "高"])
    M["short_group"] = pd.qcut(M["short_risk"].rank(method="first"), 3, labels=["低", "中", "高"])

    # 四象限(以中位數切)
    lt, stv = M["long_risk"].median(), M["short_risk"].median()
    def quad(r):
        hi_l = r["long_risk"] >= lt
        hi_s = r["short_risk"] >= stv
        if hi_l and hi_s: return "A 立即介入(長高+短高)"
        if hi_l and not hi_s: return "B 慢性追蹤(長高+短低)"
        if not hi_l and hi_s: return "C 血糖波動注意(長低+短高)"
        return "D 常規追蹤(長低+短低)"
    M["quadrant"] = M.apply(quad, axis=1)

    # 每位病患風險表
    # [修正] 新增 _fromlast 系列欄位(從最後一筆實際狀態出發),供 App「未來走勢預估」
    # 使用;hypo_{15m…24h}/hyper_{15m…24h}(固定 InRange 起點)則保留供病患風險表對照
    # 與研究內部一致性。〔稽核修正 B-15〕原寫 hypo_1h/hyper_1h,現行欄名為 15m/30m/60m/
    # 90m/4h/24h(1 小時即 60m)。
    cols = (["record", "patient", "long_risk", "long_group"]
            + [f"hypo_{h}" for h in HORIZONS] + [f"hyper_{h}" for h in HORIZONS]
            + [f"hypo_{h}_fromlast" for h in HORIZONS] + [f"hyper_{h}_fromlast" for h in HORIZONS]
            + ["last_glucose", "last_state", "short_risk"]
            + [f"short_risk_{h}" for h in HORIZONS]
            + ["short_group", "TIR", "TBR", "TAR", "quadrant", "has_complication",
               "long_risk_sd", "long_risk_lo95", "long_risk_hi95"])
    M[cols].round(3).to_csv(f"{OUTDIR}/patient_risk_table.csv", index=False, encoding="utf-8-sig")

    # 二維風險圖
    colors = {"A 立即介入(長高+短高)": "#dc2626", "B 慢性追蹤(長高+短低)": "#f59e0b",
              "C 血糖波動注意(長低+短高)": "#3b82f6", "D 常規追蹤(長低+短低)": "#16a34a"}
    plt.figure(figsize=(7.2, 6))
    for q, c in colors.items():
        sub = M[M["quadrant"] == q]
        plt.scatter(sub["long_risk"] * 100, sub["short_risk"], s=40, alpha=0.75,
                    color=c, edgecolor="white", linewidth=0.5, label=q)
    plt.axvline(lt * 100, color="grey", ls="--", lw=1)
    plt.axhline(stv, color="grey", ls="--", lw=1)
    plt.xlabel("長期惡化風險:P(任一併發症) (%)")
    _hl = {"15m": "15 分鐘", "30m": "30 分鐘", "60m": "60 分鐘", "90m": "90 分鐘",
           "4h": "4 小時", "24h": "24 小時"}.get(SHORT_HORIZON, SHORT_HORIZON)
    # [修正 B 配套] 原本軸標題寫死「24h」,改時界後會與實際數值不符
    plt.ylabel(f"短期惡化風險:未來 {_hl} 出範圍機率 (%)")
    plt.title("糖尿病惡化風險二維分層圖")
    plt.legend(fontsize=8, loc="upper left", framealpha=0.9)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/risk_map_2d.png", dpi=150); _show()

    # 象限計數 + 各象限實際併發症比例(驗證分層有效性)
    tab = M.groupby("quadrant").agg(
        人數=("record", "size"),
        平均長期風險=("long_risk", lambda x: round(x.mean() * 100, 1)),
        平均短期風險=("short_risk", lambda x: round(x.mean(), 1)),
        實際併發症比例=("has_complication", lambda x: round(x.mean() * 100, 1)),
    ).reset_index()
    tab.to_csv(f"{OUTDIR}/quadrant_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== 四象限摘要 ===")
    print(tab.to_string(index=False))

    # [修正 B] 冗餘度診斷:各時界之短期軸與「範圍外時間占比(100-TIR)」的相關
    print("\n=== 短期風險軸冗餘度診斷(與 100-TIR 之相關;越接近 1 表示未提供額外資訊)===")
    outr = 100 - M["TIR"]
    for h in HORIZONS:
        c = np.corrcoef(M[f"short_risk_{h}"], outr)[0, 1]
        print(f"  {h:>4}: corr={c:.4f}   平均絕對差={np.abs(M[f'short_risk_{h}']-outr).mean():.2f} pp")

    print(f"\n完成。輸出於 ./{OUTDIR}/")
    print("圖:risk_map_2d.png    表:patient_risk_table.csv、quadrant_summary.csv")
    # 示範:印出幾筆高風險病患
    top = M.sort_values(["long_risk", "short_risk"], ascending=False).head(5)
    print("\n最高綜合風險前 5 筆:")
    print(top[["record", "long_risk", "short_risk", "quadrant"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()