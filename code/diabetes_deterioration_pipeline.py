# -*- coding: utf-8 -*-
"""
糖尿病惡化風險預測 — 端到端可重複管線 (Shanghai_T2DM_Summary)
================================================================
本檔取代原本分散的 7 支腳本(資料轉換 / 補值 / log / 交互作用 / 標準化 /
Logistic Regression / XGBoost / KNN vs GMM),整併為單一可重複流程,並修正
以下方法學問題:

  (1) 前處理洩漏:原流程在「切分前」就對整份資料 fit 中位數與 StandardScaler,
      再做交叉驗證 → 測試折資訊外洩。本檔改把 impute + scale 放進 sklearn
      Pipeline,只在每個訓練折上 fit。
  (2) 病患層級洩漏:109 筆來自 100 位病患(8 位有多次回診)。同一人若同時落在
      train 與 test,會高估效能。本檔現行預設為 DEDUP_ONE_PER_PATIENT=False
      (保留全部 109 筆)+ USE_GROUP_CV=True,以 StratifiedGroupKFold 依病患分組
      切分;去重模式仍保留為選項。[修正 21:原文所述之「預設每位病患只留一筆」
      與現行常數設定相反,已依實際設定改寫。]
  (3) 特徵洩漏:Other Agents / Hypoglycemic Agents 這類用藥常是「因為有併發症才開」
      (如 epalrestat→神經病變、calcium dobesilate→視網膜病變),會把答案洩漏給模型。
      本檔預設不納入這些用藥二元變數(可用 INCLUDE_MED_FEATURES 開關檢視影響)。
  (4) 目標定義:橫斷面資料無法真正預測「未來惡化」。本檔提供三個目標,並新增
      複合「Any_Complication(併發症有無)」作為『疾病已惡化/已分層』的代理指標,
      同時保留大血管 / 小血管兩個細分目標。低血糖(10/109)因陽性過少不列為主要目標。
  (5) 冗餘特徵:原流程同時保留原始值與 log 值並各自標準化 → 高度共線,LR 係數不穩。
      本檔對右偏變數「只用 log 版本取代原始值」,不重複。
  (6) 修正 KNN/GMM 腳本的 bug(引用未定義的 df["Cluster"])與命名(KMeans≠KNN),
      並加入 K 值/成分數的模型選擇(silhouette / BIC)。

交叉驗證:StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED+r),
以病患 ID 為分組鍵,重複 N_REPEATS=100 次;指標取 100 次之中位數並附
2.5/97.5 百分位區間。[修正 21:原文未載明重複次數與區間口徑。]
〔稽核修正 B-15〕原寫「以 Patient Number 為分組鍵」不精確:Patient Number 本身為紀錄編號
(109 個相異值,例 2069_0_20210621、2069_1_20210705、2069_2_20210825),分組鍵實為其第一個底線前之前綴
(病患 ID,100 位;見 load_data() 之 patient_id)。

執行環境:Python 3.10+ / Colab。需要 pandas, numpy, scikit-learn, matplotlib。
xgboost 與 shap 為選用:若已安裝則使用 XGBoost + SHAP,否則自動退回
HistGradientBoosting + permutation importance(結果格式相同,可先跑通)。

作者備註:全流程固定 random_state=42;所有輸出寫入 ./output/。
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
import warnings
import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython()            # 存在於 Jupyter / Colab → 用預設 inline 後端,圖顯示在儲存格下方
    _IN_NOTEBOOK = True
except NameError:
    matplotlib.use("Agg")    # 純腳本 / 無視窗環境 → 只存檔
    _IN_NOTEBOOK = False
import matplotlib.pyplot as plt


def _show():
    """存檔後,若在 notebook 就 inline 顯示,再關閉 figure 釋放記憶體。"""
    if _IN_NOTEBOOK:
        plt.show()
    plt.close()

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedGroupKFold,
    cross_val_predict, cross_validate,
)  # 〔稽核修正 CODE-26〕刪除未使用之 GroupKFold import
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)  # [修正 9] average_precision_score/f1_score/precision_score/recall_score/accuracy_score
   # 原本手動計算用,改採 cross_validate(scoring={...}) 字串形式後未再呼叫,清除未用 import
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# 0. 設定
# ------------------------------------------------------------------ #
SEED = 42
XLSX_PATH = "Shanghai_T2DM_Summary.xlsx"     # 原始摘要檔 (sheet 名 T2DM)
OUTDIR = "output"
DEDUP_ONE_PER_PATIENT = False  # False:保留全部 109 筆,改用 StratifiedGroupKFold 依病患分組(不洩漏)  # [修正 21]
USE_GROUP_CV = True            # True:StratifiedGroupKFold(依病患 ID = Patient Number 底線前之前綴);與上面搭配使用  # [修正 21]〔稽核修正 B-15〕
INCLUDE_MED_FEATURES = False   # False:排除用藥二元變數(避免答案洩漏);設 True 可比較影響
RUN_EDA = True                 # True:輸出 EDA 圖(直方圖/箱型圖/相關矩陣/缺失熱圖/目標分布)
os.makedirs(OUTDIR, exist_ok=True)
# 〔稽核修正 CODE-26〕刪除未使用之 rng = np.random.RandomState(SEED)(全檔無任何引用;
#   其他程式亦未 import 之,刪除不影響任何亂數序列)。

# 嘗試載入 XGBoost / SHAP;沒有就退回 sklearn 內建
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# ------------------------------------------------------------------ #
# 1. 讀取與編碼(直接讀原始 xlsx,不依賴中間 CSV)
# ------------------------------------------------------------------ #
def has_event(x):
    """none / no / 空值 → 0;其餘(有記錄任何病症/用藥)→ 1"""
    if pd.isna(x):
        return 0
    x = str(x).strip().lower()
    return 0 if x in ["none", "no", "nan", ""] else 1


def load_data():
    raw = pd.read_excel(XLSX_PATH, sheet_name="T2DM")

    # 病患 ID(用於分組交叉驗證,避免同一人同時進 train/test)
    # 〔稽核修正 B-15〕= Patient Number 第一個底線前之前綴(例 2069_1_20210705 → 2069)
    # 〔稽核修正 CODE-26〕(僅註明)本資料之列序即檔名序號 _0、_1、_2;凡以
    #   duplicated(keep="first") 取「首筆」者(去重模式、成本母體加總)取到的是 _0。
    #   但 2017(_0=20210102、_1=20201118)與 2055(_0=20210524、_1=20201207)之 _0
    #   並非最早一次就診。現行結果即依此定義,未更改演算法。
    patient_id = raw["Patient Number"].astype(str).str.split("_").str[0]

    d = pd.DataFrame(index=raw.index)
    # --- 人口學 / 身體組成(不做 log)---
    d["Age"] = pd.to_numeric(raw["Age (years)"], errors="coerce")
    d["Height"] = pd.to_numeric(raw["Height (m)"], errors="coerce")
    d["Weight"] = pd.to_numeric(raw["Weight (kg)"], errors="coerce")
    d["BMI"] = pd.to_numeric(raw["BMI (kg/m2)"], errors="coerce")
    # --- 生活行為 / 病史 ---
    d["Gender"] = raw["Gender (Female=1, Male=2)"].map({1: 0, 2: 1})
    d["Smoking"] = (pd.to_numeric(raw["Smoking History (pack year)"], errors="coerce").fillna(0) > 0).astype(int)
    d["Alcohol"] = (raw["Alcohol Drinking History (drinker/non-drinker)"].astype(str)
                    .str.strip().str.lower().map({"non-drinker": 0, "drinker": 1}).fillna(0).astype(int))
    d["Duration"] = pd.to_numeric(raw["Duration of diabetes (years)"], errors="coerce")

    # --- 生化檢驗(右偏 → 稍後改用 log1p 取代原值)---
    lab = {
        "FPG": "Fasting Plasma Glucose (mg/dl)",
        "PPG": "2-hour Postprandial Plasma Glucose (mg/dl)",
        "Cpep_f": "Fasting C-peptide (nmol/L)",
        "Cpep_2h": "2-hour Postprandial C-peptide (nmol/L)",
        "Ins_f": "Fasting Insulin (pmol/L)",
        "Ins_2h": "2-hour Postprandial insulin (pmol/L)",
        "HbA1c": "HbA1c (mmol/mol)",
        "GA": "Glycated Albumin (%)",
        "TC": "Total Cholesterol (mmol/L)",
        "TG": "Triglyceride (mmol/L)",
        "HDL": "High-Density Lipoprotein Cholesterol (mmol/L)",
        "LDL": "Low-Density Lipoprotein Cholesterol (mmol/L)",
        "Cr": "Creatinine (umol/L)",
        "eGFR": "Estimated Glomerular Filtration Rate  (ml/min/1.73m2) ",
        "UA": "Uric Acid (mmol/L)",
        "BUN": "Blood Urea Nitrogen (mmol/L)",
    }
    right_skewed = ["FPG", "PPG", "Cpep_f", "Cpep_2h", "Ins_f", "Ins_2h",
                    "TG", "TC", "Cr", "eGFR", "UA", "BUN", "Duration"]
    for k, v in lab.items():
        col = pd.to_numeric(raw[v].replace("/", np.nan), errors="coerce")
        d[k] = col
    # [修正] 在 log1p 轉換「之前」留一份原始臨床單位副本,供 EDA 使用。
    #        原程式把轉換後的 d 直接傳給 run_eda(),導致 describe.csv 與所有 EDA 圖
    #        皆為 log 尺度(例如 FPG mean 顯示 5.05 而非 164.87),與報告表 4.1 對不起來。
    d_raw = d.copy()

    # 右偏變數改用 log1p 取代(HbA1c/GA/HDL/LDL 分布較對稱,保留原值)
    for k in right_skewed:
        if k in d.columns:
            # log1p 需非負;負值(理論上不會有)保留原值
            d[k] = np.where(d[k] >= 0, np.log1p(d[k]), d[k])

    # --- 可能洩漏的用藥/共病二元變數(預設不用)---
    med = pd.DataFrame(index=raw.index)
    med["Comorbidities"] = raw["Comorbidities"].apply(has_event)
    med["HypoAgents"] = raw["Hypoglycemic Agents"].apply(has_event)
    med["OtherAgents"] = raw["Other Agents"].apply(has_event)

    # --- 目標變數 ---
    Macro = raw["Diabetic Macrovascular  Complications"].apply(has_event)
    Micro = raw["Diabetic Microvascular Complications"].apply(has_event)
    Acute = raw["Acute Diabetic Complications"].apply(has_event)  # 本資料全為 none → 恆 0
    Hypo = (raw["Hypoglycemia (yes/no)"].astype(str).str.strip().str.lower()
            .map({"no": 0, "yes": 1}).fillna(0).astype(int))
    targets = pd.DataFrame({
        "Macrovascular": Macro,
        "Microvascular": Micro,
        # 複合「惡化/已分層」代理:任一種併發症(此資料 = 大血管 或 小血管)
        "Any_Complication": ((Macro + Micro + Acute) > 0).astype(int),
        "Hypoglycemia": Hypo,   # 陽性過少,僅供參考
    })

    return d, med, targets, patient_id, d_raw   # [修正] 多回傳 d_raw(原始臨床單位,供 EDA)


# ------------------------------------------------------------------ #
# 2. 建立無洩漏的 Pipeline(前處理在 CV 內完成)
# ------------------------------------------------------------------ #
def make_pipeline(kind):
    """kind: 'lr' 基準模型;'gb' 主要模型(XGBoost 或退回 HistGB)"""
    if kind == "lr":
        clf = LogisticRegression(max_iter=5000, class_weight="balanced",
                                 solver="liblinear", random_state=SEED)
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ])
    else:
        if HAS_XGB:
            clf = XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                eval_metric="logloss", random_state=SEED,
                # 〔稽核修正 A-13〕原註解「scale_pos_weight 於各折內視類別比例調整較嚴謹,
                #   此處用固定值近似」不實:本處未設定 scale_pos_weight(XGBoost 以預設 1 處理,
                #   即不做類別加權);LR 則以 class_weight="balanced" 加權。兩模型之類別權重
                #   設定因此不同,門檻相依之指標(Precision/Recall/F1)不宜直接並列比較。
                #   以上超參數為固定值,未經調參。
            )
        else:
            clf = HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=300,
                l2_regularization=1.0, random_state=SEED,
            )
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", clf),
        ])


N_REPEATS = 100   # [修正 A] 折分配重複次數(20 次不足以穩定 95% 區間尾端)


def get_cv(y, groups, use_group):
    """[修正 A] 原本使用 GroupKFold(5)(未設 shuffle),其折分配由
    np.argsort(各組樣本數) 決定;本資料 100 個病患分組中有 92 個 size=1,argsort
    預設為非穩定排序,平手順序隨 numpy 版本與 CPU 而異 → 折分配不可跨平台重現,
    且實測其預設切分對小血管/大血管兩目標落在 200 次隨機切分的 97.5 百分位之外
    (偏樂觀)。改用 StratifiedGroupKFold(shuffle=True, random_state=...) 重複
    N_REPEATS 次:random_state 真正生效、可跨平台重現,且指標可附區間。
    〔稽核修正 B-15〕原寫「GroupKFold 不接受 random_state」已過時:scikit-learn 1.6 起
    GroupKFold 亦支援 shuffle/random_state。選用 StratifiedGroupKFold 之理由在於
    可同時「依病患分組」與「依結局分層」,此理由仍成立。

    〔稽核修正 B-13〕去重模式(use_group=False)原為 RepeatedStratifiedKFold(n_repeats=10),
    共 50 折,evaluate() 之 *_rep_* 欄因「折數 ≠ N_REPEATS×5」而不產生,main() 取
    AUC_rep_median 時 KeyError,該模式實際無法執行。改為與主分析相同之 N_REPEATS 次
    重複(RepeatedStratifiedKFold 依重複次序每 5 折一組,與 evaluate() 之 reshape 相容)。
    預設模式(依病患分組)不受影響。"""
    if use_group and groups is not None:
        splits = [sp for r in range(N_REPEATS)
                  for sp in StratifiedGroupKFold(
                      n_splits=5, shuffle=True, random_state=SEED + r
                  ).split(np.zeros(len(y)), y, groups)]
        return splits, groups
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED), None


# ------------------------------------------------------------------ #
# 3. RQ2:基準(LR) vs 主要(XGBoost/GB)模型比較 + ROC + 混淆矩陣
# ------------------------------------------------------------------ #
def evaluate(X, y, groups, target_name, use_group=False):
    cv, grp = get_cv(y, groups, use_group)
    scoring = {"AUC": "roc_auc", "AP": "average_precision",
               "Accuracy": "accuracy", "Precision": "precision",
               "Recall": "recall", "F1": "f1"}
    rows = []
    for kind, label in [("lr", "Logistic Regression (baseline)"),
                        ("gb", f"{'XGBoost' if HAS_XGB else 'HistGB'} (main)")]:
        pipe = make_pipeline(kind)
        res = cross_validate(pipe, X, y, cv=cv, groups=grp, scoring=scoring,
                             error_score=np.nan)
        # 〔稽核修正 B-16〕原本 error_score=np.nan + 下方 np.nanmean/np.nanpercentile,加上檔頭
        #   全域關閉警告,任何一折配適或計分失敗都會被「靜默剔除」,結果仍照常輸出。
        #   現改為逐指標計數失敗折(分數為 NaN 者);有任何失敗即中止並說明,不輸出殘缺結果。
        #   (無失敗時下方 nan* 函式與一般函式結果相同,輸出不變。)
        n_fail = {m: int(np.isnan(np.asarray(res[f"test_{m}"], float)).sum()) for m in scoring}
        if any(n_fail.values()):
            raise RuntimeError(
                f"[{target_name} / {label}] 交叉驗證共 {len(res['test_AUC'])} 折,其中有折配適或"
                f"計分失敗(各指標失敗折數:{n_fail})。原寫法會以 np.nanmean 靜默剔除這些折;"
                f"請先查明原因(例如某折只有單一類別、資料缺欄)再重跑。")
        row = {"Target": target_name, "Model": label,
               "N": len(y), "Positives": int(y.sum()),
               "N_splits": len(res["test_AUC"])}
        for m in scoring:
            v = np.asarray(res[f"test_{m}"], float)
            # [修正 A] 單點平均不足以呈現折分配造成的不確定性。
            #   *_fold_*  : 500 個折(100 次重複 × 5 折)的分布(反映每折僅約 22 筆的抽樣雜訊,
            #              必然很寬)
            #   *_rep_*   : 先把每次重複的 5 折平均起來,再看 100 次重複的分布。
            #              後者才是「報告中那個數字換一次切分會移動多少」,應優先引用。
            #   〔稽核修正 B-15〕原註解寫「100 個折」「20 次重複」,為 [修正 A] 前之舊值;
            #   現行 N_REPEATS=100,即 500 折、100 次重複。
            row[f"{m}_mean"] = np.nanmean(v)
            row[f"{m}_sd"] = np.nanstd(v)
            row[f"{m}_fold_median"] = float(np.nanmedian(v))
            row[f"{m}_fold_lo95"] = float(np.nanpercentile(v, 2.5))
            row[f"{m}_fold_hi95"] = float(np.nanpercentile(v, 97.5))
            if len(v) == N_REPEATS * 5:
                rep = np.nanmean(v.reshape(N_REPEATS, 5), axis=1)
                row[f"{m}_rep_median"] = float(np.median(rep))
                row[f"{m}_rep_lo95"] = float(np.percentile(rep, 2.5))
                row[f"{m}_rep_hi95"] = float(np.percentile(rep, 97.5))
                row[f"{m}_rep_min"] = float(rep.min())
                row[f"{m}_rep_max"] = float(rep.max())
        rows.append(row)
    return pd.DataFrame(rows)


def roc_and_confusion(X, y, target_name, groups=None):
    """用 out-of-fold 預測畫 ROC(LR vs 主要模型)並輸出混淆矩陣。
    保留 109 筆時傳入 groups(病患 id),ROC 亦以依病患分組之折分配產生
    out-of-fold 預測。[修正 21:原文寫 GroupKFold,實際為
    StratifiedGroupKFold(5, shuffle=True, random_state=SEED)。]"""
    if USE_GROUP_CV and not DEDUP_ONE_PER_PATIENT and groups is not None:
        # [修正 A] 圖用單一固定且可重現的切分(random_state=SEED)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_groups = groups
    else:
        cv, cv_groups = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED), None
    plt.figure(figsize=(5.2, 5))
    oof = {}
    for kind, label, color in [("lr", "Logistic Regression", "#2563eb"),
                              ("gb", f"{'XGBoost' if HAS_XGB else 'HistGB'}", "#dc2626")]:
        pipe = make_pipeline(kind)
        proba = cross_val_predict(pipe, X, y, cv=cv, groups=cv_groups, method="predict_proba")[:, 1]
        oof[kind] = proba
        auc = roc_auc_score(y, proba)
        fpr, tpr, _ = roc_curve(y, proba)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {target_name}")
    plt.legend(loc="lower right", fontsize=9); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/ROC_{target_name}.png", dpi=150); _show()

    # 混淆矩陣(主要模型,門檻 0.5)
    pred = (oof["gb"] >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix — {target_name} ({'XGBoost' if HAS_XGB else 'HistGB'}, thr=0.5)")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/CM_{target_name}.png", dpi=150); _show()
    return oof


# ------------------------------------------------------------------ #
# 4. RQ1:特徵重要性(SHAP 若可用,否則 permutation importance)
#     另外輸出 LR 標準化係數 → odds ratio(可解釋)
# ------------------------------------------------------------------ #
def feature_importance(X, y, target_name):
    # (a) 主要模型的重要性
    pipe = make_pipeline("gb")
    pipe.fit(X, y)
    model = pipe.named_steps["clf"]
    X_imp = pipe.named_steps["impute"].transform(X)

    if HAS_SHAP and HAS_XGB:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_imp)
        shap.summary_plot(sv, X_imp, feature_names=list(X.columns), show=False)
        plt.title(f"SHAP Summary — {target_name}")
        plt.tight_layout(); plt.savefig(f"{OUTDIR}/SHAP_{target_name}.png", dpi=150); _show()
        imp = pd.DataFrame({"Feature": X.columns,
                            "Importance": np.abs(sv).mean(axis=0)})
    else:
        r = permutation_importance(pipe, X, y, scoring="roc_auc",
                                   n_repeats=20, random_state=SEED)
        imp = pd.DataFrame({"Feature": X.columns,
                            "Importance": r.importances_mean})
    imp = imp.sort_values("Importance", ascending=False).reset_index(drop=True)

    # 條狀圖(前 15,標數值)
    top = imp.head(15).iloc[::-1]
    plt.figure(figsize=(7, 5))
    bars = plt.barh(top["Feature"], top["Importance"], color="#0d9488")
    xmax = float(top["Importance"].max())
    for b, v in zip(bars, top["Importance"]):
        plt.text(b.get_width() + xmax * 0.01, b.get_y() + b.get_height() / 2,
                 f"{v:.3f}", va="center", fontsize=8)
    plt.margins(x=0.12)
    plt.title(f"Feature Importance (top 15) — {target_name}")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/IMP_{target_name}.png", dpi=150); _show()

    # (b) LR 係數 → odds ratio(給統計系報告用,直觀可解釋)
    lr = make_pipeline("lr"); lr.fit(X, y)
    coef = lr.named_steps["clf"].coef_[0]
    oddsr = pd.DataFrame({"Feature": X.columns,
                          "LR_coef(std)": coef,
                          "OddsRatio": np.exp(coef)}).sort_values(
                          "LR_coef(std)", key=np.abs, ascending=False).reset_index(drop=True)
    return imp, oddsr


# ------------------------------------------------------------------ #
# 5. RQ3:分群(KMeans 選 K + GMM 選成分數) + PCA 圖 + 卡方檢定
# ------------------------------------------------------------------ #
def clustering(X, targets, cluster_cols):
    Xc = SimpleImputer(strategy="median").fit_transform(X[cluster_cols])
    Xc = StandardScaler().fit_transform(Xc)

    # (a) KMeans:用 silhouette 選 K(2..6)
    sil = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xc)
        sil[k] = silhouette_score(Xc, km.labels_)
    best_k = max(sil, key=sil.get)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Xc)
    km_lab = km.labels_

    # (b) GMM:用 BIC 選成分數(2..6)
    bic = {}
    for k in range(2, 7):
        gm = GaussianMixture(n_components=k, random_state=SEED).fit(Xc)
        bic[k] = gm.bic(Xc)
    best_g = min(bic, key=bic.get)
    gm = GaussianMixture(n_components=best_g, random_state=SEED).fit(Xc)
    gm_lab = gm.predict(Xc)

    # (c) PCA 2D 視覺化
    pca = PCA(n_components=2, random_state=SEED)
    xy = pca.fit_transform(Xc)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    ax[0].scatter(xy[:, 0], xy[:, 1], c=km_lab, cmap="viridis", s=25)
    ax[0].set_title(f"K-means (k={best_k}, silhouette={sil[best_k]:.3f})")
    ax[1].scatter(xy[:, 0], xy[:, 1], c=gm_lab, cmap="viridis", s=25)
    ax[1].set_title(f"GMM (k={best_g}, BIC-selected)")
    for a in ax:
        a.set_xlabel("PC1"); a.set_ylabel("PC2")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/Clustering_PCA.png", dpi=150); _show()

    # (d) 卡方:各分群 vs 併發症比例
    lines = []
    prop_rows = []   # [修正 10] 原本 prop(各群併發症比例)算出即捨棄,報告因此只有
                      # p 值、沒有效果方向與大小(哪一群較高、高多少),此處補存供引用。
    for lab, name in [(km_lab, f"KMeans(k={best_k})"), (gm_lab, f"GMM(k={best_g})")]:
        for t in ["Microvascular", "Macrovascular", "Any_Complication"]:
            tab = pd.crosstab(lab, targets[t])
            chi2, p, dof, _ = chi2_contingency(tab)
            prop = pd.crosstab(lab, targets[t], normalize="index").mul(100).round(1)
            lines.append(f"[{name}] {t}: chi2={chi2:.2f}, df={dof}, p={p:.4f}")
            for grp_label in prop.index:
                prop_rows.append({"method": name, "target": t, "cluster": grp_label,
                                   "pct_positive": float(prop.loc[grp_label, 1]) if 1 in prop.columns else np.nan})
    pd.DataFrame(prop_rows).to_csv(f"{OUTDIR}/cluster_complication_rate.csv",
                                    index=False, encoding="utf-8-sig")
    return {"silhouette_by_k": sil, "bic_by_k": bic,
            "best_k": best_k, "best_gmm": best_g, "chi2": lines}


# ------------------------------------------------------------------ #
# 5b. EDA:直方圖 / 箱型圖 / 相關矩陣 / 缺失熱圖 / 目標分布
# ------------------------------------------------------------------ #
def run_eda(X, targets):
    edir = os.path.join(OUTDIR, "eda_figures")
    os.makedirs(edir, exist_ok=True)

    # (1) 目標變數分布(檢查類別不平衡)
    tt = ["Any_Complication", "Microvascular", "Macrovascular", "Hypoglycemia"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    for ax, t in zip(axes, tt):
        vc = targets[t].value_counts().sort_index()
        vals = [int(vc.get(0, 0)), int(vc.get(1, 0))]
        bars = ax.bar(["No", "Yes"], vals, color=["#94a3b8", "#ef4444"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v),
                    ha="center", va="bottom", fontsize=10)
        ax.margins(y=0.15)
        ax.set_title(f"{t}\n(pos={int(targets[t].sum())}/{len(targets)})", fontsize=9)
    fig.suptitle("Target distribution", y=1.03)
    plt.tight_layout(); plt.savefig(f"{edir}/00_target_distribution.png", dpi=150, bbox_inches="tight"); _show()

    # (2) 缺失值熱圖 + 缺失比例
    plt.figure(figsize=(9, 4))
    import seaborn as sns
    sns.heatmap(X.isna(), cbar=False, cmap="Greys")
    plt.title("Missing-value map (white=missing)"); plt.xlabel("features"); plt.ylabel("records")
    plt.tight_layout(); plt.savefig(f"{edir}/01_missing_heatmap.png", dpi=150); _show()
    # 〔稽核修正 CODE-23〕原 sort_values 未指定排序法(預設 quicksort,非穩定),同分之特徵
    #   (8 個 0.0%、21.1% 之 Cr/GA、16.5% 之 HDL/TG 等)之先後隨平台而異(Windows 與 Linux
    #   實測順序不同)。改用穩定排序 mergesort:同分者一律依原欄位順序,跨平台固定。數值不變。
    miss = (X.isna().mean() * 100).round(1).sort_values(ascending=False, kind="mergesort")
    # [修正 19] 原輸出之索引與數值欄皆無名稱(標頭為 ",0"),讀檔者無從得知
    #   第 2 欄之單位是「百分比」還是「比例」。補上欄名與索引名。
    miss.rename("missing_pct").to_csv(f"{edir}/missing_pct.csv",
                                      index_label="feature", encoding="utf-8-sig")

    # (3) 連續變數直方圖(網格)
    cont = [c for c in X.columns if X[c].nunique() > 5]
    ncol = 4; nrow = int(np.ceil(len(cont) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 2.3))
    for ax, c in zip(axes.ravel(), cont):
        vals = X[c].dropna()
        ax.hist(vals, bins=20, color="#3b82f6", edgecolor="white")
        mu, med = vals.mean(), vals.median()
        ax.axvline(mu, color="#dc2626", lw=1.2)
        ax.set_title(f"{c}\nmean={mu:.1f} / median={med:.1f}", fontsize=7.5)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(cont):]:
        ax.axis("off")
    fig.suptitle("Histograms of continuous features", y=1.005)
    plt.tight_layout(); plt.savefig(f"{edir}/02_histograms.png", dpi=150, bbox_inches="tight"); _show()

    # (4) 箱型圖(標準化後,方便並排看離群值)
    Xz = (X[cont] - X[cont].mean()) / X[cont].std()
    plt.figure(figsize=(min(1 + 0.55 * len(cont), 15), 4.5))
    Xz.boxplot(rot=90, grid=False)
    # 在每個箱子上方標「原始尺度中位數」
    for i, c in enumerate(cont, start=1):
        med = X[c].median()
        plt.text(i, Xz[c].quantile(0.75) + 0.15, f"{med:.1f}",
                 ha="center", va="bottom", fontsize=6, color="#dc2626", rotation=90)
    plt.title("Box plots (z-scored; red = raw-scale median) — outlier check")
    plt.tight_layout()
    plt.savefig(f"{edir}/03_boxplots.png", dpi=150); _show()

    # (5) 相關矩陣(連續變數,格子標數值)
    corr = X[cont].corr()
    plt.figure(figsize=(14, 12))
    # 〔稽核修正 S4〕原圖每格皆標兩位小數、字級 6,縮入報告後約 1.6 pt 無法辨讀。改為只標 |r| >= 0.5
    #   之格(字級 10),其餘以顏色呈現;數值全表仍輸出於 correlation_matrix.csv。
    lab = corr.map(lambda v: f"{v:.2f}" if abs(v) >= 0.5 else "")
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
                annot=lab, fmt="", annot_kws={"size": 10},
                linewidths=0.3, cbar_kws={"shrink": 0.7})
    plt.title("Correlation matrix (continuous features; |r| >= 0.5 labelled)")
    plt.tight_layout(); plt.savefig(f"{edir}/04_correlation_matrix.png", dpi=150); _show()
    corr.to_csv(f"{edir}/correlation_matrix.csv",
                index_label="feature", encoding="utf-8-sig")   # [修正 19]

    # (6) 敘述統計表(mean/sd/偏態/缺失比例)
    desc = X.describe().T
    desc["skew"] = X.skew(numeric_only=True)
    desc["missing_%"] = X.isna().mean() * 100
    desc.round(3).to_csv(f"{edir}/describe.csv",
                         index_label="feature", encoding="utf-8-sig")   # [修正 19]
    print(f"[EDA] 圖表已輸出至 {edir}/(目標分布、缺失熱圖、直方圖、箱型圖、相關矩陣、敘述統計)\n")


# ------------------------------------------------------------------ #
# 6. 主程式
# ------------------------------------------------------------------ #
def main():
    print(f"[env] XGBoost={'yes' if HAS_XGB else 'no (→HistGB)'} | "
          f"SHAP={'yes' if HAS_SHAP else 'no (→permutation)'}\n")

    d, med, targets, pid, d_raw = load_data()

    # 特徵集合(預設:純臨床變數,不含用藥洩漏變數)
    feature_cols = list(d.columns)
    if INCLUDE_MED_FEATURES:
        d = pd.concat([d, med], axis=1)
        feature_cols += list(med.columns)

    # 去重(每位病患留第一筆),避免病患層級洩漏
    # [修正 4] d_raw 原本未隨 d/targets/pid 同步去重:DEDUP_ONE_PER_PATIENT=True 時
    #          d 會變成 100 筆但 d_raw 仍是 109 筆,兩者以不同列數傳入 run_eda(),
    #          會讓「目標分布」(依 targets,100 筆)與缺失熱圖/直方圖/相關矩陣
    #          (依 d_raw,109 筆)靜默地對不上,不會報錯但 EDA 圖表彼此矛盾。
    if DEDUP_ONE_PER_PATIENT:
        keep = ~pid.duplicated(keep="first").values
        d, targets, pid = d[keep].reset_index(drop=True), targets[keep].reset_index(drop=True), pid[keep].reset_index(drop=True)
        d_raw = d_raw[keep].reset_index(drop=True)
    unit = "位病患" if DEDUP_ONE_PER_PATIENT else "筆紀錄"
    # [修正 18] 原字串寫 "GroupKFold(依病患)",但 get_cv() 自 [修正 A] 起已改用
    #   StratifiedGroupKFold(shuffle=True, random_state=SEED+r) 重複 N_REPEATS 次。
    #   印出之 CV 名稱與實際不符,會誤導讀者以為折分配不可重現之舊作法仍在使用。
    cvname = (f"StratifiedGroupKFold(5, shuffle=True, random_state=SEED+r) × {N_REPEATS} 次重複，依病患分組"
              if (USE_GROUP_CV and not DEDUP_ONE_PER_PATIENT)
              else f"RepeatedStratifiedKFold(5 折 × {N_REPEATS} 重複)")   # 〔稽核修正 B-13〕
    print(f"分析樣本:N={len(d)} {unit}(病患 {pid.nunique()} 位),特徵數={len(feature_cols)},CV={cvname}\n")

    X = d[feature_cols].copy()

    # EDA:直方圖 / 箱型圖 / 相關矩陣 / 缺失熱圖 / 目標分布
    if RUN_EDA:
        # [修正] 傳原始臨床單位而非 log1p 後的建模資料
        run_eda(d_raw.reindex(columns=[c for c in feature_cols if c in d_raw.columns]), targets)

    # 主要建模目標(低血糖陽性太少,不列入主要比較)
    main_targets = ["Any_Complication", "Microvascular", "Macrovascular"]

    all_metrics = []
    for t in main_targets:
        y = targets[t].astype(int)
        print(f"===== 目標:{t}(陽性 {int(y.sum())}/{len(y)})=====")
        # 保留 109 筆時用 StratifiedGroupKFold(依 pid 分組,shuffle,重複 N_REPEATS 次);
        # 去重時用 RepeatedStratifiedKFold。[修正 21]
        m = evaluate(X, y, pid, t, use_group=(USE_GROUP_CV and not DEDUP_ONE_PER_PATIENT))
        all_metrics.append(m)
        print(m[["Model", "AUC_rep_median", "AUC_rep_lo95", "AUC_rep_hi95", "AP_rep_median", "F1_rep_median", "Recall_rep_median"]].round(3).to_string(index=False))
        roc_and_confusion(X, y, t, groups=pid)
        imp, oddsr = feature_importance(X, y, t)
        imp.to_csv(f"{OUTDIR}/importance_{t}.csv", index=False, encoding="utf-8-sig")
        oddsr.to_csv(f"{OUTDIR}/lr_oddsratio_{t}.csv", index=False, encoding="utf-8-sig")
        print("Top5 重要特徵:", ", ".join(imp["Feature"].head(5)), "\n")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df.to_csv(f"{OUTDIR}/model_results.csv", index=False, encoding="utf-8-sig")

    # RQ3 分群(用核心臨床變數)
    cluster_cols = ["Age", "BMI", "Duration", "HbA1c", "FPG", "TG", "HDL", "LDL", "Cr", "eGFR"]
    cinfo = clustering(X, targets, cluster_cols)
    print("===== RQ3 分群 =====")
    print("KMeans silhouette by k:", {k: round(v, 3) for k, v in cinfo["silhouette_by_k"].items()})
    # [修正 29] 分群之模型選擇指標原本只印在畫面上,沒有任何輸出檔。報告第肆章
    #   第五節引用的「Silhouette 係數僅 0.149」因此無從覆核 —— 四層驗證都看不到。
    #   此處存成 CSV。
    pd.DataFrame(
        [{"method": "KMeans", "k": k, "criterion": "silhouette", "value": v,
          "selected": k == cinfo.get("best_k")} for k, v in cinfo["silhouette_by_k"].items()]
        + [{"method": "GMM", "k": k, "criterion": "BIC", "value": v,
            "selected": k == cinfo.get("best_gmm")} for k, v in cinfo["bic_by_k"].items()]
    ).to_csv(f"{OUTDIR}/cluster_selection.csv", index=False, encoding="utf-8-sig")
    print(f"[cluster] 模型選擇指標 → {OUTDIR}/cluster_selection.csv")
    print("最佳 K =", cinfo["best_k"], "| GMM 最佳成分數 =", cinfo["best_gmm"])
    for l in cinfo["chi2"]:
        print("  ", l)

    print(f"\n完成。所有結果與圖表已輸出至 ./{OUTDIR}/")
    print("圖:ROC_*.png、CM_*.png、IMP_*.png(或 SHAP_*.png)、Clustering_PCA.png")
    print("表:model_results.csv、importance_*.csv、lr_oddsratio_*.csv")


if __name__ == "__main__":
    main()