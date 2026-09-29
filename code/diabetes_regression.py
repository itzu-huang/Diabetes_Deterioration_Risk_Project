# -*- coding: utf-8 -*-
"""
迴歸問題:以臨床特徵預測 HbA1c(長期血糖控制)
====================================================
對應作業 3.5「迴歸問題:基準模型 Linear Regression + 主要模型 XGBoost/RF」,
評估指標為 MAE、RMSE、MAPE、R²(作業 3.6)。與分類流程共用同一套資料處理
(直接 import diabetes_deterioration_pipeline.load_data),並同樣以「依病患分組」之交叉驗證
  [修正 21:原文寫 GroupKFold;實作為隨機分派病患至 5 折 × 100 次重複]
依病患分組交叉驗證,避免病患層級洩漏。

目標可替換:改 TARGET 即可預測其他連續變數(如 FPG、BMI);若要預測 CGM 衍生的
TIR / GMI,見檔尾 REG_ON_CGM 說明(會把 cgm_metrics.csv 併回)。

環境:需要 pandas, numpy, scikit-learn, matplotlib;xgboost 選用(無則退回
HistGradientBoostingRegressor)。與 diabetes_deterioration_pipeline.py 放同目錄。
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
import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython()            # Jupyter / Colab / Spyder → inline 顯示
    _NB = True
except NameError:
    matplotlib.use("Agg")
    _NB = False
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import PredefinedSplit, cross_val_predict   # 〔稽核修正 CODE-26〕刪除未使用之 GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor

from diabetes_deterioration_pipeline import load_data   # 重用同一套資料處理

SEED = 42
OUTDIR = "output_reg"
os.makedirs(OUTDIR, exist_ok=True)

TARGET = "HbA1c"                       # 迴歸目標(連續變數);可改成 "FPG"、"BMI" 等
EXCLUDE_AS_FEATURE = ["HbA1c", "GA"]   # 排除目標與其近似指標(糖化白蛋白 GA 與 HbA1c r≈0.85)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def _show():
    if _NB:
        plt.show()
    plt.close()


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y != 0
    return np.mean(np.abs((y[m] - p[m]) / y[m])) * 100


def make_reg(kind):
    """kind='lin' 基準(Linear Regression);'gb' 主要(XGBoost 或退回 HistGB)。"""
    if kind == "lin":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("reg", LinearRegression())])
    if HAS_XGB:
        reg = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                           random_state=SEED)
    else:
        reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                            max_iter=300, l2_regularization=1.0,
                                            random_state=SEED)
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("reg", reg)])


def main():
    print(f"[env] XGBoost={'yes' if HAS_XGB else 'no (→HistGB)'}")
    d, med, targets, pid, _ = load_data()   # load_data 現多回傳 d_raw(EDA 用)

    y = d[TARGET].astype(float)
    feats = [c for c in d.columns if c not in EXCLUDE_AS_FEATURE]
    X = d[feats].copy()
    keep = y.notna().values
    X, y, g = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True), pid[keep].reset_index(drop=True)
    # [修正 18] 原印出 "GroupKFold(依病患)",但 [修正 A] 起已改為「隨機把病患
    #   分派到 5 折」(fold_assign + PredefinedSplit)重複 100 次。名稱須與實作一致。
    print(f"迴歸目標 = {TARGET} | N = {len(y)} | 特徵數 = {len(feats)} | "
          f"CV = 隨機分派病患至 5 折(PredefinedSplit) × 100 次重複，依病患分組\n")

    # [修正 A] 原本使用 GroupKFold(5)(未設 shuffle),折分配由 argsort
    #          的平手順序決定,不可跨平台重現;實測 R2 在不同折分配下由 -0.03 到 0.33。
    #          〔稽核修正 B-15〕原寫「該類別不接受 random_state」已過時(scikit-learn 1.6 起
    #          GroupKFold 支援 shuffle/random_state);改採下列作法之理由(可重現、可附區間)不變。
    #          改為「隨機把病患分派到 5 折」重複 N_REPEATS 次,報告中位數與 95% 區間。
    N_REPEATS = 100
    pids = np.array(sorted(set(g)))

    def fold_assign(seed):
        rs = np.random.RandomState(seed)
        perm = rs.permutation(len(pids))
        m = {pids[j]: k % 5 for k, j in enumerate(perm)}
        return np.array([m[p] for p in g])

    rows, oof = [], {}
    main_label = f"{'XGBoost' if HAS_XGB else 'HistGB'} (main)"
    for kind, label, color in [("lin", "Linear Regression (baseline)", "#2563eb"),
                              ("gb", main_label, "#dc2626")]:
        mae_l, rmse_l, mape_l, r2_l = [], [], [], []
        for r in range(N_REPEATS):
            fa = fold_assign(SEED + r)
            p_ = cross_val_predict(make_reg(kind), X, y, cv=PredefinedSplit(fa))
            if r == 0:
                oof[kind] = p_          # 圖用第一次(可重現)之 out-of-fold 預測
            mae_l.append(mean_absolute_error(y, p_))
            rmse_l.append(float(np.sqrt(mean_squared_error(y, p_))))
            mape_l.append(mape(y, p_))
            r2_l.append(r2_score(y, p_))
        # [修正 20] reg_results.csv 原僅有 Model 一欄可辨識,未記錄迴歸目標、
        #   樣本數與特徵數。同一檔名在不同 TARGET 下會被覆寫且無從分辨。
        d_ = dict(Target=TARGET, N=len(y), n_features=len(feats),
                  Model=label, N_repeats=N_REPEATS)
        for nm, v in [("MAE", mae_l), ("RMSE", rmse_l), ("MAPE", mape_l), ("R2", r2_l)]:
            v = np.array(v)
            d_[f"{nm}_median"] = float(np.median(v))
            d_[f"{nm}_lo95"] = float(np.percentile(v, 2.5))
            d_[f"{nm}_hi95"] = float(np.percentile(v, 97.5))
            d_[f"{nm}_min"] = float(v.min()); d_[f"{nm}_max"] = float(v.max())
        rows.append(d_)
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUTDIR}/reg_results.csv", index=False, encoding="utf-8-sig")
    print(R.round(3).to_string(index=False))

    # (1) 實際 vs 預測 散佈圖(兩模型)
    lo, hi = float(y.min()), float(y.max())
    plt.figure(figsize=(5.4, 5.2))
    plt.scatter(y, oof["lin"], s=22, alpha=0.6, color="#2563eb", label="Linear Regression")
    plt.scatter(y, oof["gb"], s=22, alpha=0.6, color="#dc2626", label=main_label.split()[0])
    plt.plot([lo, hi], [lo, hi], "--", color="grey", lw=1, label="perfect")
    plt.xlabel(f"Actual {TARGET}"); plt.ylabel(f"Predicted {TARGET}")
    plt.title(f"Predicted vs Actual — {TARGET}")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/reg_pred_vs_actual_{TARGET}.png", dpi=150); _show()

    # (2) 殘差圖(主要模型)
    resid = y.values - oof["gb"]
    plt.figure(figsize=(6, 3.4))
    plt.scatter(oof["gb"], resid, s=20, alpha=0.6, color="#0d9488")
    plt.axhline(0, color="grey", ls="--", lw=1)
    plt.xlabel(f"Predicted {TARGET}"); plt.ylabel("Residual (actual − pred)")
    plt.title(f"Residuals — {main_label.split()[0]}")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/reg_residuals_{TARGET}.png", dpi=150); _show()

    # (3) 特徵重要性(permutation,對主要模型)
    pipe = make_reg("gb"); pipe.fit(X, y)
    r = permutation_importance(pipe, X, y, scoring="r2", n_repeats=20, random_state=SEED)
    imp = pd.DataFrame({"Feature": X.columns, "Importance": r.importances_mean}) \
        .sort_values("Importance", ascending=False).reset_index(drop=True)
    imp.to_csv(f"{OUTDIR}/reg_importance_{TARGET}.csv", index=False, encoding="utf-8-sig")
    top = imp.head(12).iloc[::-1]
    plt.figure(figsize=(7, 4.5))
    bars = plt.barh(top["Feature"], top["Importance"], color="#7c3aed")
    xm = float(top["Importance"].max()) if top["Importance"].max() > 0 else 1
    for b, v in zip(bars, top["Importance"]):
        plt.text(b.get_width() + xm * 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                 va="center", fontsize=8)
    plt.margins(x=0.14)
    plt.title(f"Permutation importance (top 12) — predicting {TARGET}")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/reg_importance_{TARGET}.png", dpi=150); _show()

    print(f"\n完成。輸出於 ./{OUTDIR}/")
    print("圖:reg_pred_vs_actual_*.png、reg_residuals_*.png、reg_importance_*.png")
    print("表:reg_results.csv、reg_importance_*.csv")
    print(f"Top5 影響 {TARGET} 的特徵:", ", ".join(imp['Feature'].head(5)))


# -------------------------------------------------------------------- #
# 想改成預測 CGM 的 TIR / GMI(把血糖控制品質當迴歸目標)?
# 1) 先跑 cgm_lstm_markov.py 產生 cgm_output/cgm_metrics.csv
# 2) 用 Patient_Number 併回 load_data 的資料(病患編號一致),把 TIR/GMI 當 y
# 3) 設 TARGET="TIR"、EXCLUDE_AS_FEATURE=[]（TIR 不在臨床特徵內,無需排除）
# -------------------------------------------------------------------- #

if __name__ == "__main__":
    main()