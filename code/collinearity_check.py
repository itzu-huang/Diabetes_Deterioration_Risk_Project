# -*- coding: utf-8 -*-
"""
collinearity_check.py —— 共線性診斷(VIF、判定係數、拔靴係數方向、剔除特徵之 AUC)
================================================================
〔稽核修正 B-10〕新增。報告 ¶521、¶802、¶803(第肆章第一節;第伍章研究限制第(十二)項)
所引用之共線性數字(VIF、R²、200 次拔靴、剔除特徵 AUC)原本在重現包中沒有任何產生程式。
本檔依報告之描述實作,並與主程式 diabetes_deterioration_pipeline.py 完全同口徑:
  ・特徵矩陣:load_data() 之 24 個建模特徵(13 項 log1p 轉換後),109 筆紀錄;
  ・插補:中位數(與主程式 Pipeline 內之 SimpleImputer(strategy="median") 相同);
  ・模型:make_pipeline("lr")(中位數插補 → 標準化 → LR;class_weight="balanced"、liblinear、C=1);
  ・目標:Any_Complication;分組:load_data() 之病患 ID(Patient Number 底線前之前綴)。

輸出(皆寫入 output/,皆為確定性結果、全精度):
  collinearity_vif.csv
      section = VIF            各特徵之 VIF = 1/(1−R²),R² 為以其餘 23 個特徵做 OLS 之判定係數
                               (全資料、中位數插補後之特徵矩陣);
      section = VIF_summary    VIF < 5 之特徵數、除身高/體重/BMI 外之最大 VIF;
      section = pearson_r      報告 ¶521 所引之兩兩相關(原始臨床單位 d_raw、成對完整觀測;
                               與 output/eda_figures/correlation_matrix.csv 同口徑);
      section = BMI_definition BMI 與 體重/身高² 之相關係數與最大絕對偏差(¶521、¶802)。
  collinearity_bootstrap_signs.csv
      200 次拔靴,每次以同一 LR Pipeline 配適後記錄各特徵係數之正負;resample_unit = record
      (逐筆重抽,np.random.RandomState(42),報告採用)與 patient(以病患為單位重抽,
      RandomState(42),處理 8 位多次回診之敏感度檢查)。minority_count = min(正, 負) 次數;
      另列「VIF<5 之特徵」之少數方向次數中位數。
  collinearity_ablation_auc.csv
      剔除特徵後之判別效能:30 組 StratifiedGroupKFold(5, shuffle=True, random_state=42+r)
      (r = 0…29,即主分析前 30 次重複之折分配),每組以全部 109 筆之 out-of-fold 機率計算
      一個 AUC,報告取 30 個 AUC 之中位數(此 AUC 定義與表 4-3-1 之「5 折平均之中位數」不同)。

預期結果(與稽核時獨立重算者相同;報告以三位小數或整數呈現):
  VIF 體重 260.07、BMI 147.83、身高 107.19、Cr 15.77、eGFR 14.67、TC 9.27;R² 0.9962/0.9932/0.9907;
  VIF<5 者 16 個特徵;拔靴少數方向次數 體重 95、BMI 73、身高 98(以病患重抽 92/71/92),
  VIF<5 之 16 個特徵中位數 11.5;
  剔除特徵之 AUC 中位數:全部 0.8420、剔除 BMI 0.8437、剔除身高與體重 0.8460、再剔除 eGFR 0.8487;
  累計剔除(BMI、身高、體重)0.8502、再剔除 eGFR 0.8510。
  ※ 報告 ¶521「遠高於前述各組之最大值(總膽固醇 9.3)」不成立:Cr、eGFR 之 VIF 為 15.77、
    14.67,亦逾 10(稽核 A-02)。
執行時間:約 1 分鐘。由 run_all.py --with-sensitivity 呼叫(亦可單獨執行)。
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

import diabetes_deterioration_pipeline as P   # 同一套 load_data() 與 make_pipeline()

OUT = "output"
SEED = P.SEED            # 42
N_BOOT = 200             # 拔靴次數(報告 ¶802)
N_ABL = 30               # 剔除特徵比較之重複切分組數(報告 ¶803)
TARGET = "Any_Complication"
# 報告 ¶521 所列之兩兩相關(原始臨床單位)
PAIRS = [("eGFR", "Cr"), ("HbA1c", "GA"), ("TC", "LDL"), ("FPG", "PPG"),
         ("Ins_f", "Ins_2h"), ("Weight", "BMI"), ("BMI", "Height")]
# 報告 ¶803 之剔除組合(依報告敘述之順序)
ABLATIONS = [
    ("全部 24 特徵", []),
    ("剔除 BMI", ["BMI"]),
    ("剔除 身高、體重", ["Height", "Weight"]),
    ("剔除 身高、體重、eGFR", ["Height", "Weight", "eGFR"]),
    ("累計剔除 BMI、身高、體重", ["BMI", "Height", "Weight"]),
    ("累計剔除 BMI、身高、體重、eGFR", ["BMI", "Height", "Weight", "eGFR"]),
]


def vif_table(X):
    """各特徵以其餘特徵做 OLS 之 R² 與 VIF = 1/(1−R²)(全資料;中位數插補)。"""
    Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                      columns=X.columns, index=X.index)
    rows = []
    for c in Xi.columns:
        others = Xi.drop(columns=[c])
        r2 = LinearRegression().fit(others, Xi[c]).score(others, Xi[c])
        rows.append(dict(feature=c, r2=float(r2), vif=float(1.0 / (1.0 - r2))))
    return pd.DataFrame(rows)


def bootstrap_signs(X, y, groups, unit):
    """N_BOOT 次拔靴之 LR 係數正負次數。unit = "record"(逐筆)或 "patient"(以病患為單位)。
    每種 unit 各自以 RandomState(SEED) 起始。任何一次配適失敗即拋出例外(不靜默略過)。"""
    rng = np.random.RandomState(SEED)
    upid = np.unique(groups)
    signs = np.empty((N_BOOT, X.shape[1]))
    for b in range(N_BOOT):
        if unit == "record":
            idx = rng.randint(0, len(y), len(y))
        else:
            pk = rng.choice(upid, len(upid), replace=True)
            idx = np.concatenate([np.where(groups == p)[0] for p in pk])
        m = P.make_pipeline("lr").fit(X.iloc[idx], y[idx])
        signs[b] = np.sign(m.named_steps["clf"].coef_[0])
    return (signs > 0).sum(0), (signs < 0).sum(0), (signs == 0).sum(0)


def oof_aucs(X, y, groups, cols):
    """N_ABL 組 StratifiedGroupKFold(5, shuffle=True, random_state=SEED+r) 之 OOF AUC。"""
    out = []
    for r in range(N_ABL):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)
        p = cross_val_predict(P.make_pipeline("lr"), X[cols], y, cv=cv, groups=groups,
                              method="predict_proba")[:, 1]
        out.append(roc_auc_score(y, p))
    return np.array(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    d, med, targets, pid, d_raw = P.load_data()
    X = d.copy()                                   # 24 個建模特徵(與主分析相同)
    y = targets[TARGET].astype(int).values
    groups = pid.values
    print(f"[collinearity] 特徵 {X.shape[1]} 個、紀錄 {len(y)} 筆、病患 {len(np.unique(groups))} 位;"
          f"目標 {TARGET}(陽性 {int(y.sum())})")

    # ---- 1. VIF、相關係數、BMI 定義式 ----------------------------------------
    V = vif_table(X)
    vif = dict(zip(V.feature, V.vif))
    rows = [dict(section="VIF", feature=r.feature, feature2="", value=r.vif, r2=r.r2, n=len(X),
                 how="1/(1−R²);R² 為以其餘特徵 OLS 迴歸之判定係數(24 特徵、log1p 後、中位數插補)")
            for r in V.itertuples()]
    lt5 = V[V.vif < 5]
    whb = {"Weight", "BMI", "Height"}
    rest = V[~V.feature.isin(whb)].sort_values("vif", ascending=False, kind="mergesort").iloc[0]
    rows.append(dict(section="VIF_summary", feature="n_features_VIF_lt5", feature2="",
                     value=float(len(lt5)), r2=np.nan, n=len(X), how="VIF < 5 之特徵數"))
    rows.append(dict(section="VIF_summary", feature="max_VIF_excl_Weight_BMI_Height", feature2=rest.feature,
                     value=rest.vif, r2=rest.r2, n=len(X),
                     how="除身高、體重、BMI 外之最大 VIF(feature2 為該特徵)"))
    for a, b in PAIRS:
        ok = d_raw[a].notna() & d_raw[b].notna()
        rows.append(dict(section="pearson_r", feature=a, feature2=b,
                         value=float(d_raw[[a, b]].corr().iloc[0, 1]), r2=np.nan, n=int(ok.sum()),
                         how="Pearson 相關;原始臨床單位、成對完整觀測(同 correlation_matrix.csv)"))
    calc = d_raw["Weight"] / d_raw["Height"] ** 2
    ok = d_raw["BMI"].notna() & calc.notna()
    rows.append(dict(section="BMI_definition", feature="BMI", feature2="Weight/Height^2",
                     value=float(np.corrcoef(d_raw.loc[ok, "BMI"], calc[ok])[0, 1]), r2=np.nan,
                     n=int(ok.sum()), how="BMI 與 體重÷身高² 之 Pearson 相關(定義式依存)"))
    rows.append(dict(section="BMI_definition", feature="max_abs_dev", feature2="BMI − Weight/Height^2",
                     value=float((d_raw.loc[ok, "BMI"] - calc[ok]).abs().max()), r2=np.nan,
                     n=int(ok.sum()), how="|BMI − 體重÷身高²| 之最大值(kg/m²)"))
    pd.DataFrame(rows).to_csv(f"{OUT}/collinearity_vif.csv", index=False, encoding="utf-8-sig")

    # ---- 2. 拔靴係數方向 --------------------------------------------------------
    brow = []
    for unit, note in [("record", "逐筆重抽(報告採用)"), ("patient", "以病患為單位重抽(敏感度)")]:
        pos, neg, zer = bootstrap_signs(X, y, groups, unit)
        mino = np.minimum(pos, neg)
        for j, c in enumerate(X.columns):
            brow.append(dict(resample_unit=unit, feature=c, VIF=vif[c], VIF_lt5=bool(vif[c] < 5),
                             n_boot=N_BOOT, n_positive=int(pos[j]), n_negative=int(neg[j]),
                             n_zero=int(zer[j]), minority_count=float(mino[j]),
                             note=f"{note};np.random.RandomState({SEED})"))
        med_lt5 = float(np.median([mino[j] for j, c in enumerate(X.columns) if vif[c] < 5]))
        brow.append(dict(resample_unit=unit, feature="median_VIF_lt5", VIF=np.nan, VIF_lt5=True,
                         n_boot=N_BOOT, n_positive=np.nan, n_negative=np.nan, n_zero=np.nan,
                         minority_count=med_lt5,
                         note=f"VIF<5 之 {len(lt5)} 個特徵之少數方向次數中位數"))
    B = pd.DataFrame(brow)
    B.to_csv(f"{OUT}/collinearity_bootstrap_signs.csv", index=False, encoding="utf-8-sig")

    # ---- 3. 剔除特徵之 OOF AUC ---------------------------------------------------
    arow = []
    for name, drop in ABLATIONS:
        cols = [c for c in X.columns if c not in drop]
        a = oof_aucs(X, y, groups, cols)
        arow.append(dict(feature_set=name, removed="、".join(drop) if drop else "(無)",
                         n_features=len(cols), n_repeats=N_ABL,
                         auc_median=float(np.median(a)), auc_mean=float(a.mean()),
                         auc_min=float(a.min()), auc_max=float(a.max()),
                         cv=f"StratifiedGroupKFold(5, shuffle=True, random_state={SEED}+r), r=0…{N_ABL-1};"
                            f"每組以 109 筆 OOF 機率計 1 個 AUC"))
    A = pd.DataFrame(arow)
    A.to_csv(f"{OUT}/collinearity_ablation_auc.csv", index=False, encoding="utf-8-sig")

    # ---- 摘要 ----------------------------------------------------------------------
    print("\n=== VIF(最高 8 個)===")
    print(V.sort_values("vif", ascending=False, kind="mergesort").head(8)
          .to_string(index=False, formatters={"r2": "{:.4f}".format, "vif": "{:.2f}".format}))
    print(f"VIF<5 之特徵數:{len(lt5)};除身高/體重/BMI 外之最大 VIF:{rest.feature} {rest.vif:.2f}")
    for unit in ["record", "patient"]:
        s = B[B.resample_unit == unit].set_index("feature").minority_count
        print(f"[拔靴 {unit}] 少數方向次數 體重 {s['Weight']:.0f}、BMI {s['BMI']:.0f}、身高 {s['Height']:.0f};"
              f"VIF<5 特徵之中位數 {s['median_VIF_lt5']:g}")
    print("\n=== 剔除特徵之 OOF AUC(30 組之中位數)===")
    print(A[["feature_set", "n_features", "auc_median"]].to_string(
        index=False, formatters={"auc_median": "{:.4f}".format}))
    print(f"\n輸出 → {OUT}/collinearity_vif.csv、collinearity_bootstrap_signs.csv、"
          f"collinearity_ablation_auc.csv")


if __name__ == "__main__":
    main()
