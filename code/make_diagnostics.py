# -*- coding: utf-8 -*-
"""[新增] 診斷數值輸出 —— 讓報告中引用但未存於既有 CSV 的數字可被覆核。
輸出 output/diagnostics_v2.csv,每列一個數值,含名稱、值、計算方式。

〔稽核修正 B-12〕原本有數列是把報告中的數字直接手輸寫進本檔(例如 lstm_params_approx=16900、
  陳興寶 2003 成本常數另抄一份、van Doorn 2021 文獻值),驗證第 3 層再以「報告數字 ← 本檔」
  當作強證據,形成循環論證。現改為:
  (1) 新增欄位 source_type,標明每一列之來源性質:
        computed      由研究資料(或其上游輸出檔)經程式計算,或由程式實際建立之模型計算而得;
                      重跑即可覆核。
        code_constant 由程式碼中之設定常數(非研究資料、非文獻對照值)經確定性運算而得,
                      例如 CPI 逐年環比指數之累計倍數;重跑只能覆核算術,不能證明常數本身正確。
        literature    文獻報告值(陳興寶 2003 表 1、van Doorn 2021)及其單位換算/比值;
                      僅供對照,不是本研究之結果,亦不能作為「報告數字已被重現」之證據。
  (2) LSTM 參數量改由模型架構實算(cgm_lstm_markov.build_lstm_model().count_params();
      未安裝 tensorflow 時以解析公式計算並交叉核對):16,961,取代手寫之 16,900。
  (3) 成本常數改由 lifetime_cost_montecarlo.COST 匯入,不再另抄一份。
〔稽核修正 A-19〕原 add() 一律 round(,4) 後寫檔,報告再從 4 位數進位即可能雙重進位;
  改為全精度寫出(各項數值本身不變)。lstm_by_length_group.csv 同。
〔稽核修正 CODE-26〕cost_gini 之勞倫茲曲線改為自原點起積分(差約 1.2×10⁻⁵)。"""

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
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import diabetes_deterioration_pipeline as P
import cgm_lstm_markov as C
import lifetime_cost_montecarlo as LC   # 〔稽核修正 B-12〕成本常數與 CPI 由此匯入(原於檔中另抄一份)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss

SOURCE_TYPES = ("computed", "code_constant", "literature")   # 〔稽核修正 B-12〕見檔頭說明
rows = []
def add(k, v, how, source_type="computed"):
    # 〔稽核修正 A-19〕原為 value=round(float(v), 4):改為全精度。
    # 〔稽核修正 B-12〕新增 source_type 欄(computed/code_constant/literature)。
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"source_type 必須為 {SOURCE_TYPES} 之一:{source_type}")
    rows.append(dict(item=k, value=float(v), how=how, source_type=source_type))

d, med, targets, pid, d_raw = P.load_data()
y = targets["Any_Complication"].astype(int)
lr = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
               ("c", LogisticRegression(max_iter=5000, class_weight="balanced",
                                        solver="liblinear", random_state=P.SEED))])
lr.fit(d, y); ins = lr.predict_proba(d)[:, 1]
M = np.empty((P.N_REPEATS, len(y)))
for r in range(P.N_REPEATS):
    M[r] = cross_val_predict(lr, d, y, cv=StratifiedGroupKFold(5, shuffle=True,
            random_state=P.SEED + r), groups=pid, method="predict_proba")[:, 1]
oof = M.mean(axis=0)
add("apparent_AUC", roc_auc_score(y, ins), "全資料配適之 in-sample AUC(任一併發症)")
add("apparent_Brier", brier_score_loss(y, ins), "同上之 Brier")
add("validated_AUC", roc_auc_score(y, oof), "交叉驗證 out-of-fold 平均機率之 AUC")
add("validated_Brier", brier_score_loss(y, oof), "同上之 Brier")
add("insample_median", np.median(ins), "in-sample 機率中位數(App 原門檻)")
add("oof_median", np.median(oof), "out-of-fold 機率中位數(App 新門檻)")
add("flagged_pct_old_cut", (oof >= np.median(ins)).mean() * 100, "以原門檻套用於 CV 機率之高風險比例(%)")

R = pd.read_csv("output_risk/patient_risk_table.csv")
outr = 100 - R["TIR"]
for h in ["15m", "30m", "60m", "90m", "4h", "24h"]:
    add(f"corr_shortrisk_{h}_vs_100minusTIR", np.corrcoef(R[f"short_risk_{h}"], outr)[0, 1],
        f"短期風險({h})與範圍外時間占比之相關")

rows_w = []
for fp in C.list_cgm_files():
    s = C.load_one(fp)
    span = (s.ts.max() - s.ts.min()).total_seconds() / 86400
    # [修正 16] 有效佩戴率分母。15 分鐘取樣、跨距 span 天之「應有時槽數」
    #   為 span*96 + 1(首尾兩端皆計入,閉區間),原式 span*96 少算 1 格,
    #   使佩戴率系統性高估。實測原式下 109 筆中有 104 筆 > 100%
    #   (最高 100.4065%),此為分母錯誤之直接證據——佩戴率在定義上不可能超過 100%。
    #   改正後範圍 99.4485–100.0000%,超過 100% 者 0 筆。
    slots = span * 96 + 1
    rows_w.append(dict(days=span,
                       active=100 * len(s) / max(slots, 1),
                       active_oldden=100 * len(s) / max(span * 96, 1)))
W = pd.DataFrame(rows_w)
add("cgm_days_median", W.days.median(), "CGM 監測天數中位數")
add("cgm_days_min", W.days.min(), "最短監測天數")
add("cgm_days_max", W.days.max(), "最長監測天數")
add("cgm_readings_raw", sum(len(pd.read_excel(f)) for f in C.list_cgm_files()), "CGM 原始列數")
add("cgm_readings_clean", sum(len(C.load_one(f)) for f in C.list_cgm_files()), "CGM 去空值後讀數")
add("cgm_n_under14d", (W.days < 14).sum(), "未達 14 天之監測筆數")
add("cgm_n_under5d", (W.days < 5).sum(), "不足 5 天之監測筆數")
add("wear_pct_min", W.active.min(), "有效佩戴率最小值(%);分母=span*96+1")
add("wear_pct_max", W.active.max(), "有效佩戴率最大值(%);分母=span*96+1")
add("wear_pct_median", W.active.median(), "有效佩戴率中位數(%);分母=span*96+1")
add("wear_n_over100", (W.active > 100).sum(), "佩戴率 >100% 之筆數(正確分母下應為 0)")
add("wear_pct_max_oldden", W.active_oldden.max(), "[對照] 舊分母 span*96 之最大值(%)——>100% 即為分母錯誤之證據")
add("wear_n_over100_oldden", (W.active_oldden > 100).sum(), "[對照] 舊分母下 >100% 之筆數")

C2 = pd.read_csv("output_cost_mc/patient_lifetime_cost_mc.csv")
f = C2[C2.counted_in_population]
cc = np.sort(f.lifetime_mean.values); n = len(cc)
for k in [10, 20, 30, 50]:
    add(f"cost_top{k}pct_share", cc[::-1][:round(n * k / 100)].sum() / cc.sum() * 100,
        f"最貴前 {k}% 病患占全體餘生總成本之比例(%)")
# 〔稽核修正 CODE-26〕勞倫茲曲線自原點 (0,0) 起積分(原式漏第一個梯形,差約 1.2×10⁻⁵)
add("cost_gini", 1 - 2 * np.trapezoid(np.r_[0.0, np.cumsum(cc) / cc.sum()],
                                      np.r_[0.0, np.arange(1, n + 1) / n]),
    "吉尼係數(勞倫茲曲線自原點起以梯形法積分)")
for q in [75, 90]:
    add(f"cost_p{q}", np.percentile(f.lifetime_mean, q), f"餘生成本第 {q} 百分位(元)")
add("cost_max", f.lifetime_mean.max(), "餘生成本最大值(元)")
add("cost_max_over_min", f.lifetime_mean.max() / f.lifetime_mean.min(), "最高/最低倍數")

A = pd.read_csv("cgm_output/lstm_vs_baseline_all.csv")
Pp = pd.read_csv("cgm_output/lstm_pooled_vs_baseline.csv")
Mg = A.merge(Pp[["record", "skill_pct"]].rename(columns={"skill_pct": "pooled"}), on="record") \
      .merge(R[["record", "has_complication"]], on="record")
for lab, v in [("with", 1), ("without", 0)]:
    sub = Mg[Mg.has_complication == v]
    add(f"lstm_per_skill_{lab}_comp", sub.skill_pct.median(), f"逐人版改善中位數({lab} complication)")
    add(f"lstm_pooled_skill_{lab}_comp", sub.pooled.median(), f"合併版改善中位數({lab} complication)")
add("lstm_gap_per", Mg[Mg.has_complication == 0].skill_pct.median()
    - Mg[Mg.has_complication == 1].skill_pct.median(), "逐人版族群落差(百分點)")
add("lstm_gap_pooled", Mg[Mg.has_complication == 0].pooled.median()
    - Mg[Mg.has_complication == 1].pooled.median(), "合併版族群落差(百分點)")

# [修正 17] 報告 4.4 節之「監測長度」診斷原本沒有任何輸出檔可供覆核
#   (含 809 / 1,175 / p<0.001 / −0.403 / 由 −0.55 降至 −0.27 五個數字),
#   四層驗證因而完全看不到。此處全部補上,並修正偏相關之值:
#   實算 Pearson 零階 −0.5487、控制監測點數後 −0.2616(→ −0.26),
#   報告所載之 −0.27 與任一合理口徑皆不符(控制 log n 為 −0.2802、
#   等級偏相關為 −0.3005),應改為 −0.26 並註明控制方式。
import scipy.stats as _st
_cgm = pd.read_csv("cgm_output/cgm_metrics.csv").rename(columns={"Patient_Number": "record"})
_D = Mg.merge(_cgm[["record", "n"]], on="record")
_hc = _D.has_complication.values.astype(float)
_sk = _D.skill_pct.values
_np_ = _D["n"].values.astype(float)


def _partial(x, y, z):
    """控制 z 後 x 與 y 之偏相關(以 OLS 殘差相關實作,等價於標準定義)。"""
    Z = np.column_stack([np.ones(len(z)), np.asarray(z, float)])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1]


add("npoints_mean_with_comp", _np_[_hc == 1].mean(), "有併發症者之平均監測點數")
add("npoints_mean_without_comp", _np_[_hc == 0].mean(), "無併發症者之平均監測點數")
_pv = float(_st.ttest_ind(_np_[_hc == 1], _np_[_hc == 0], equal_var=False).pvalue)
add("npoints_ttest_p", _pv, "兩組監測點數之 Welch t 檢定 p 值")
# 原 add() 一律 round(,4),極小的 p 會被四捨五入成 0.0000 而失去資訊,
# 故另存 -log10(p):7.04 即 p≈9.0e-08。(〔稽核修正 A-19〕add() 已改為全精度,本列保留以維持相容。)
add("npoints_ttest_neg_log10_p", -np.log10(_pv) if _pv > 0 else 99.0,
    "同上之 -log10(p);7.04 表示 p≈9.0e-08(報告所述 p<0.001 成立)")
_raw_hb = pd.read_excel("Shanghai_T2DM_Summary.xlsx", sheet_name="T2DM")
_hb = _D.merge(_raw_hb[["Patient Number", "HbA1c (mmol/mol)"]]
               .rename(columns={"Patient Number": "record"}), on="record")
_hb["hb"] = pd.to_numeric(_hb["HbA1c (mmol/mol)"], errors="coerce")
_m = _hb.hb.notna()
add("corr_npoints_hba1c", np.corrcoef(_hb.loc[_m, "n"], _hb.loc[_m, "hb"])[0, 1],
    f"監測點數與糖化血色素之 Pearson 相關(n={int(_m.sum())};8 筆 HbA1c 缺值已排除)")
add("corr_comp_skill_zero_order", np.corrcoef(_hc, _sk)[0, 1],
    "併發症與逐人版改善幅度之零階相關(點二系列=Pearson)")
add("partial_comp_skill_ctrl_npoints", _partial(_hc, _sk, _np_),
    "★ 控制監測點數(線性)後之偏相關——報告應引用此值")
add("partial_comp_skill_ctrl_log_npoints", _partial(_hc, _sk, np.log(_np_)),
    "[對照] 改以 log(監測點數) 為控制變項")
add("partial_comp_skill_rank", _partial(_st.rankdata(_hc), _st.rankdata(_sk), _st.rankdata(_np_)),
    "[對照] 等級(Spearman 型)偏相關")
add("skill_median_with_comp_pp", _D[_D.has_complication == 1].skill_pct.median(),
    "有併發症者之逐人版改善中位數(%)")
add("skill_median_without_comp_pp", _D[_D.has_complication == 0].skill_pct.median(),
    "無併發症者之逐人版改善中位數(%)")
add("beatpct_with_comp", (_D[_D.has_complication == 1].skill_pct > 0).mean() * 100,
    "有併發症者之逐人版勝率(%)")
add("beatpct_without_comp", (_D[_D.has_complication == 0].skill_pct > 0).mean() * 100,
    "無併發症者之逐人版勝率(%)")
# 〔稽核修正 CODE-26/CODE-12〕原為「先 add 佔位值(lstm_corr_per_npoints)、隨即刪除」之無作用
#   程式碼;報告 ¶648「改善幅度與監測序列長度高度相關(r ≈ +0.77)」因此沒有任何輸出可覆核。
#   改為實算。項名採 lstm_per_skill_ 開頭,與其他 LSTM 衍生項一致(verify_outputs.py 據此
#   將之歸為 LSTM 相依級)。
add("lstm_per_skill_corr_npoints", np.corrcoef(_sk, _np_)[0, 1],
    "逐人版改善幅度(skill_pct)與監測點數之 Pearson 相關(報告 ¶648 之 r)")

# 成本參數之衍生值(陳興寶 2003 表 1)
# 〔稽核修正 B-12〕改由 lifetime_cost_montecarlo.COST 匯入(原於此另抄一份常數);
#   此組為文獻值,source_type = literature。
COST_D = LC.COST["direct"]
COST_T = LC.COST["total"]
for k,v in COST_D.items(): add(f"cost_direct_{k}", v, "陳興寶 2003 表 1:年直接醫療費用(2001 RMB);取自 lifetime_cost_montecarlo.COST", "literature")
for k,v in COST_T.items(): add(f"cost_total_{k}", v, "陳興寶 2003 表 1:年總費用(2001 RMB);取自 lifetime_cost_montecarlo.COST", "literature")
add("cost_ratio_both_vs_none_direct", COST_D["both"]/COST_D["none"], "大+小 / 無(直接費用)倍數;由文獻值計算", "literature")
add("cost_ratio_both_vs_none_total", COST_T["both"]/COST_T["none"], "大+小 / 無(總費用)倍數;由文獻值計算", "literature")
# CPI 累計倍數逐年(〔稽核修正 B-12〕由程式常數 MEDICAL_CPI 連乘而得 → code_constant)
mult=1.0
for yr in range(LC.COST_BASE_YEAR+1, LC.TARGET_YEAR+1):
    mult *= LC.MEDICAL_CPI[yr]/100.0
    add(f"cpi_cum_{yr}", mult, "上海醫療保健類 CPI 自 2001 之累計倍數(lifetime_cost_montecarlo.MEDICAL_CPI 連乘)", "code_constant")


def _lstm_param_counts(look_back=12, units=64):
    """〔稽核修正 B-12〕LSTM 參數量由模型架構實算:以 cgm_lstm_markov.build_lstm_model()
    (全專案 LSTM 之唯一架構定義)建立模型後 count_params();並以解析公式交叉核對
    (LSTM 層 = 4 閘 ×(輸入權重 + 遞迴權重 + 偏差)= 4×(units×(1+units)+units);Dense(1) = units+1)。
    未安裝 tensorflow 時退回解析公式。回傳 (LSTM 層, 總數, 計算方式)。"""
    ana_layer = 4 * (units * (1 + units) + units)
    ana_total = ana_layer + units * 1 + 1
    try:
        import tensorflow  # noqa: F401  只確認可否載入;不設種子(不影響本檔其他計算)
    except ImportError:
        return ana_layer, ana_total, "解析公式(未安裝 tensorflow)"
    m = C.build_lstm_model(look_back, 1, units)
    layer, total = int(m.layers[0].count_params()), int(m.count_params())
    if (layer, total) != (ana_layer, ana_total):
        raise RuntimeError(f"LSTM 參數量 keras={layer}/{total} 與解析公式 {ana_layer}/{ana_total} 不符")
    return layer, total, "keras count_params(),與解析公式相符"


_pl, _pt, _how = _lstm_param_counts()
add("lstm_params_layer", _pl, f"LSTM(64 單元)層之參數量;{_how}")
add("lstm_params_total", _pt, f"含 Dense 輸出層之總參數量;{_how}")
# 〔稽核修正 B-12〕原為手輸之 16900(「報告所述之約略值」),使第 3 層以本檔覆核報告之 16,900
#   成為循環論證。改為由模型實算之總參數量(16,961);報告宜寫 16,961 或「約 17,000」。
add("lstm_params_approx", _pt, f"模型實算之總參數量(原為手輸之 16900;報告宜改寫為 16,961 或約 17,000);{_how}")

# 文獻對照值(van Doorn 等, PLoS ONE 2021;16(6):e0253125),原文以 mmol/L 報告
# 〔稽核修正 B-12〕以下皆為文獻值(及其換算/比值),source_type = literature
MMOL = 18.018
LIT = {"vandoorn_all_15m":0.19,"vandoorn_all_60m":0.59,
       "vandoorn_t2d_15m":0.29,"vandoorn_t2d_60m":0.70,
       "vandoorn_t1d_15m":0.43,"vandoorn_t1d_60m":1.73}
for k,v in LIT.items():
    add(k+"_mmolL", v, "van Doorn 2021 原文報告值(mmol/L)", "literature")
    add(k+"_mgdL", v*MMOL, "同上換算為 mg/dL(×18.018)", "literature")
add("vandoorn_t1d_over_t2d_60m", LIT["vandoorn_t1d_60m"]/LIT["vandoorn_t2d_60m"],
    "第 1 型 / 第 2 型之 60 分鐘 RMSE 倍數;由文獻值計算", "literature")
add("vandoorn_n_train", 851, "van Doorn 2021 之納入人數", "literature")

# [修正 29] 報告表 4-7-7(依監測長度四分位分組之效能)與「合併訓練共 88,481 個
#   視窗」原本沒有任何輸出檔可對應,四層驗證完全看不到。此處補上。
_Pp = pd.read_csv("cgm_output/lstm_pooled_vs_baseline.csv")
add("pooled_train_windows_total", _Pp.n_train_windows.sum(),
    "跨病患合併訓練之訓練視窗總數(= 各筆 n_train_windows 之和)")
add("pooled_points_mean", _Pp.n_points.mean(), "各筆監測點數之平均")
_LG = A.merge(_Pp[["record", "n_points", "skill_pct"]]
              .rename(columns={"skill_pct": "pooled"}), on="record")
_q = pd.qcut(_LG.n_points, 4, labels=["最短 25%", "短", "長", "最長 25%"])
_G = _LG.groupby(_q, observed=True).agg(
    n=("record", "size"), mean_points=("n_points", "mean"),
    per_skill_median=("skill_pct", "median"), pooled_skill_median=("pooled", "median"))
_G["per_beat_pct"] = _LG.groupby(_q, observed=True).apply(
    lambda s: (s.skill_pct > 0).mean() * 100, include_groups=False)
_G["pooled_beat_pct"] = _LG.groupby(_q, observed=True).apply(
    lambda s: (s.pooled > 0).mean() * 100, include_groups=False)
_G.to_csv("output/lstm_by_length_group.csv",   # 〔稽核修正 A-19〕原 .round(4) 後寫檔,改全精度
          index_label="length_group", encoding="utf-8-sig")
print("[diag] 依監測長度分組之效能 → output/lstm_by_length_group.csv")

pd.DataFrame(rows).to_csv("output/diagnostics_v2.csv", index=False, encoding="utf-8-sig")
print(f"已輸出 {len(rows)} 項 → output/diagnostics_v2.csv")
print(pd.DataFrame(rows).to_string(index=False))
