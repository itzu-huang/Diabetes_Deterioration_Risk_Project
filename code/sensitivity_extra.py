# -*- coding: utf-8 -*-
"""
sensitivity_extra.py —— 〔稽核修正 A-14／A-15／A-21／A-26〕補做之敏感度與假設檢查
================================================================
稽核發現報告有四項「應做而未做」之分析。本檔一次補齊,結果寫入 output/,報告依此改寫:

  (一) A-26 共病／用藥變數:主分析排除 Comorbidities、Hypoglycemic Agents、Other Agents。
       用藥之排除理由(因病情開立→答案洩漏)成立;共病多為高血壓、高血脂等風險因子,
       「因病情開立」之理由不適用。此處比較 納入共病、納入共病＋兩項用藥 之 AUC。
  (二) A-21 對數轉換之選擇:主分析之 13 項 log1p 為設計時依變數性質(濃度型檢驗值＋罹病年數)
       指定,並非逐一依樣本偏態判定(總膽固醇偏態僅 0.25 卻被轉換;高密度脂蛋白 1.72、
       身體質量指數 1.12 未轉換)。此處比較:完全不轉換、改依樣本偏態(|偏態|>1)決定轉換。
  (三) A-15 成本參數:成本變異係數 COST_CV(0.15／0.30／0.45)與折現率(0%／3%／5%)
       對每人成本、象限平均與高低風險倍數((A+B)/(C+D))之影響。
  (四) A-14 一階馬可夫假設:以 109 個 CGM 檔之實際「兩步(30 分鐘)轉移」與一階模型
       之 P² 比對,並量化二階相依性(前一狀態是否影響下一步)。

  另輸出(五)表 4-7-3 之「實測時間占比」兩種口徑(109 筆逐筆平均 vs 全部讀值合併)。

設計原則:
  - (一)(二)只用 Logistic Regression(報告之主模型;確定性,跨作業系統結果相同),
    交叉驗證與主分析完全相同:StratifiedGroupKFold(5, shuffle=True, random_state=SEED+r),
    r=0..N_REPEATS−1,每次重複取 5 折平均 AUC,報告 100 次之中位數與 2.5/97.5 百分位。
    「主分析」列必須與 output/model_results.csv 之 LR 列逐位元相同(程式自動核對)。
  - (三)之基準情境(CV 0.30、折現 3%)必須重現 output_cost_mc/ 之既有結果(自動核對),
    其餘情境使用同一亂數種子(共同隨機數),差異只來自參數。
  - (四)之狀態定義、15 分鐘時步、30 分鐘斷點規則與 deterioration_risk.py 完全相同。

輸出(5 個 CSV):
  output/sensitivity_features_transform.csv   (一)(二)各特徵集／轉換方式之 AUC
  output/sensitivity_skewness_table.csv       (二)各連續特徵之樣本偏態與是否被轉換
  output/sensitivity_cost_params.csv          (三)
  output/markov_order_check.csv               (四)逐筆
  output/markov_order_summary.csv             (四)(五)彙總
"""
# ---------------------------------------------------------------------------
# [修正 32] Windows 主控台編碼防護(同其他程式)。
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
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

import diabetes_deterioration_pipeline as P
import lifetime_cost_montecarlo as LC
import deterioration_risk as DR

OUT = "output"
os.makedirs(OUT, exist_ok=True)
N_REPEATS = P.N_REPEATS                      # 100,與主分析相同
TARGETS = ["Any_Complication", "Microvascular", "Macrovascular"]


# ============================================================ (一)(二) 特徵與轉換
def cv_auc_lr(X, y, pid):
    """與 sens_missing.py／主分析同一折分配之 LR 交叉驗證 AUC(每次重複取 5 折平均)。"""
    v = []
    for r in range(N_REPEATS):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=P.SEED + r)
        res = cross_validate(P.make_pipeline("lr"), X, y, cv=cv, groups=pid,
                             scoring={"AUC": "roc_auc"}, error_score="raise")
        v.append(float(np.mean(res["test_AUC"])))
    v = np.array(v)
    return float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def feature_transform_sensitivity():
    d, med, targets, pid, d_raw = P.load_data()
    cont = [c for c in d_raw.columns if c not in ("Gender", "Smoking", "Alcohol")]
    skew = {c: float(stats.skew(d_raw[c].dropna(), bias=False)) for c in cont}
    main_logged = [c for c in d.columns if not np.allclose(
        d[c].fillna(-999).values, d_raw[c].fillna(-999).values)]
    by_skew = sorted([c for c in cont if abs(skew[c]) > 1], key=lambda c: -abs(skew[c]))

    def log_cols(frame, cols):
        f = frame.copy()
        for c in cols:
            f[c] = np.where(f[c] >= 0, np.log1p(f[c]), f[c])
        return f

    sets = {
        "主分析(24 特徵;13 項依設計 log1p)": d,
        "加入共病 Comorbidities(25)": pd.concat([d, med[["Comorbidities"]]], axis=1),
        "加入共病＋兩項用藥(27;用藥有洩漏疑慮,僅供對照)": pd.concat([d, med], axis=1),
        "不做對數轉換(24)": d_raw.copy(),
        f"改依樣本偏態 |偏態|>1 轉換({len(by_skew)} 項)": log_cols(d_raw, by_skew),
    }
    rows = []
    for name, X in sets.items():
        for t in TARGETS:
            y = targets[t].astype(int)
            med_, lo, hi = cv_auc_lr(X, y, pid)
            rows.append(dict(analysis=name, n_features=X.shape[1], target=t, model="LR",
                             AUC_median=med_, AUC_lo95=lo, AUC_hi95=hi))
            print(f"  {name:<40s} {t:<17s} {med_:.4f} [{lo:.4f}, {hi:.4f}]")
    S = pd.DataFrame(rows)
    S.to_csv(f"{OUT}/sensitivity_features_transform.csv", index=False, encoding="utf-8-sig")

    # 主分析列須與 model_results.csv 之 LR 列逐位元相同
    mr = os.path.join(OUT, "model_results.csv")
    if os.path.exists(mr):
        MR = pd.read_csv(mr, float_precision="round_trip")
        MR = MR[MR.Model == "Logistic Regression (baseline)"]
        base = S[S.analysis.str.startswith("主分析")]
        for _, x in base.iterrows():
            m = MR[MR.Target == x.target].iloc[0]
            if float(m.AUC_rep_median) != x.AUC_median:
                raise SystemExit(f"★ 主分析列與 model_results.csv 不一致:{x.target} "
                                 f"{x.AUC_median} vs {m.AUC_rep_median}")
        print("  ✓ 主分析列與 model_results.csv 之 LR AUC_rep_median 逐位元相同")
    # 偏態資訊一併存檔,供報告改寫 A-21 之依據
    sk = pd.DataFrame({"feature": list(skew), "skewness": list(skew.values()),
                       "log1p_in_main": [c in main_logged for c in skew],
                       "abs_skew_gt_1": [abs(skew[c]) > 1 for c in skew]})
    sk.sort_values("skewness", ascending=False, kind="mergesort").to_csv(
        f"{OUT}/sensitivity_skewness_table.csv", index=False, encoding="utf-8-sig")
    return S, sk


# ============================================================ (三) 成本參數
def cost_param_sensitivity():
    d, med, targets, pid, _ = LC.load_data()
    raw = pd.read_excel("Shanghai_T2DM_Summary.xlsx", sheet_name="T2DM")
    rec = raw["Patient Number"].astype(str).values
    quad = LC._require_risk_table(rec)
    age = d["Age"].fillna(d["Age"].median()).values
    gender = d["Gender"].values
    is_first = ~pd.Series(pid.values).duplicated(keep="first").values
    pm = LC.oof_prob(d, targets["Macrovascular"].astype(int), pid, class_weight=LC.CLASS_WEIGHT)
    pi = LC.oof_prob(d, targets["Microvascular"].astype(int), pid, class_weight=LC.CLASS_WEIGHT)
    c = LC.COST[LC.COST_TYPE]

    def run(cost_cv, disc):
        # 與 lifetime_cost_montecarlo.main() 相同之抽樣順序(基準情境可逐位元重現)
        rg = np.random.default_rng(LC.SEED)
        k = lambda mean, cv, n: (np.full(n, mean, float) if cv <= 0
                                 else rg.gamma(1.0 / cv ** 2, mean * cv ** 2, size=n))
        cost_draw = {kk: k(v, cost_cv, LC.N_SIM) for kk, v in c.items()}
        infl = k(LC.INFLATION_MEAN, LC.INFLATION_CV, LC.N_SIM)
        am = LC._surv_annuity_table(LC._l_male, disc)
        af = LC._surv_annuity_table(LC._l_female, disc)
        per = np.empty(len(d)); pop = np.zeros(LC.N_SIM)
        for i in range(len(d)):
            tab = af if gender[i] == 0 else am if gender[i] == 1 else (am + af) / 2
            ann = float(np.interp(age[i], np.arange(len(tab)), tab))
            mac = rg.random(LC.N_SIM) < pm[i]
            mic = rg.random(LC.N_SIM) < pi[i]
            cs = np.select([mac & mic, mac & ~mic, ~mac & mic],
                           [cost_draw["both"], cost_draw["macro"], cost_draw["micro"]],
                           default=cost_draw["none"])
            lt = cs * infl * ann
            per[i] = lt.mean()
            if is_first[i]:
                pop += lt
        f = is_first
        q = np.array([str(x)[0] for x in quad])
        qm = {Q: per[f & (q == Q)].mean() for Q in "ABCD"}
        hi = per[f & np.isin(q, ["A", "B"])].mean(); lo = per[f & np.isin(q, ["C", "D"])].mean()
        s = np.sort(per[f])[::-1]; top20 = s[:int(round(0.2 * len(s)))].sum() / s.sum()
        return dict(COST_CV=cost_cv, discount_rate=disc,
                    mean_per_patient=per[f].mean(), median_per_patient=float(np.median(per[f])),
                    pop_total_mean=pop.mean(), pop_lo95=float(np.percentile(pop, 2.5)),
                    pop_hi95=float(np.percentile(pop, 97.5)),
                    quad_A=qm["A"], quad_B=qm["B"], quad_C=qm["C"], quad_D=qm["D"],
                    ratio_AB_over_CD=hi / lo, top20_share=top20)

    scen = [(0.30, 0.03), (0.15, 0.03), (0.45, 0.03), (0.30, 0.00), (0.30, 0.05)]
    S = pd.DataFrame([run(cv, r) for cv, r in scen])
    S.insert(0, "scenario", ["基準(CV 0.30、折現 3%)", "CV 0.15", "CV 0.45", "折現 0%", "折現 5%"])
    S.to_csv(f"{OUT}/sensitivity_cost_params.csv", index=False, encoding="utf-8-sig")

    # 基準情境須重現既有輸出(每人平均、象限平均)
    qci = pd.read_csv("output_cost_mc/quadrant_cost_interval.csv")
    b = S.iloc[0]
    for Q in "ABCD":
        ref = float(qci.loc[qci.iloc[:, 0].astype(str).str[0] == Q, "mean"].iloc[0])
        if abs(ref - b[f"quad_{Q}"]) > 0.5:
            raise SystemExit(f"★ 基準情境未能重現 quadrant_cost_interval.csv:{Q} {b[f'quad_{Q}']} vs {ref}")
    print("  ✓ 基準情境之四象限平均與 quadrant_cost_interval.csv 相同")
    with pd.option_context("display.width", 200):
        print(S[["scenario", "mean_per_patient", "median_per_patient", "quad_A", "quad_B",
                 "quad_C", "quad_D", "ratio_AB_over_CD", "top20_share"]].round(3).to_string(index=False))
    return S


# ============================================================ (四)(五) 馬可夫一階假設
def markov_order_check():
    files = sorted(glob.glob(os.path.join(DR.CGM_DIR, "*.xlsx")) +
                   glob.glob(os.path.join(DR.CGM_DIR, "*.xls")))
    rt = pd.read_csv("output_risk/patient_risk_table.csv")
    rt = rt.set_index("record")
    rows = []
    # 合併計數:二階 N[a,b,c](三個連續讀值、兩個間隔皆 <=30 分)
    N3 = np.zeros((3, 3, 3))
    pooled_counts = np.zeros(3)
    for fp in files:
        rid = os.path.splitext(os.path.basename(fp))[0]
        s = DR.load_cgm(fp)
        g = s["cgm"].values
        if len(g) < 20:
            continue
        st = np.array([DR.st3(v) for v in g])
        pooled_counts += np.bincount(st, minlength=3)
        gap = pd.Series(s["ts"].values).diff().dt.total_seconds().div(60).values
        ok1 = np.isfinite(gap) & (gap <= DR.MAX_GAP_MIN)          # ok1[i]: i-1→i 有效
        Pm = DR.transition_matrix(list(st), ts=s["ts"].values)
        model_30 = (np.linalg.matrix_power(Pm, 2)[1, 0] + np.linalg.matrix_power(Pm, 2)[1, 2]) * 100
        # 實際兩步:起點在範圍內、t→t+1、t+1→t+2 皆有效
        n2 = 0; out2 = 0
        for i in range(len(st) - 2):
            if st[i] == 1 and ok1[i + 1] and ok1[i + 2]:
                n2 += 1; out2 += st[i + 2] != 1
            if ok1[i + 1] and ok1[i + 2]:
                N3[st[i], st[i + 1], st[i + 2]] += 1
        emp_30 = out2 / n2 * 100 if n2 else np.nan
        rows.append(dict(record=rid, n_two_step_from_inrange=n2,
                         model_short_risk_30m=model_30, empirical_out_of_range_30m=emp_30,
                         csv_short_risk_30m=float(rt.loc[rid, "short_risk_30m"])
                         if rid in rt.index else np.nan))
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/markov_order_check.csv", index=False, encoding="utf-8-sig")

    # 以兩種短期軸各自之中位數切分,比較分組是否改變
    mh = R.model_short_risk_30m >= R.model_short_risk_30m.median()
    eh = R.empirical_out_of_range_30m >= R.empirical_out_of_range_30m.median()
    rho, p_rho = stats.spearmanr(R.model_short_risk_30m, R.empirical_out_of_range_30m)
    # 二階相依:目前在範圍內時,下一步出範圍之機率,依「前一狀態」分
    def p_out_next(prev):
        n = N3[prev, 1, :].sum()
        return (N3[prev, 1, 0] + N3[prev, 1, 2]) / n * 100 if n else np.nan, int(n)
    p_prev_in, n_prev_in = p_out_next(1)
    p_prev_low, n_prev_low = p_out_next(0)
    p_prev_high, n_prev_high = p_out_next(2)
    # 概似比檢定:二階 vs 一階(合併資料)
    G2 = 0.0; df = 0
    for b in range(3):
        n_b = N3[:, b, :].sum(axis=0)                 # 一階:b→c
        if n_b.sum() == 0:
            continue
        p1 = n_b / n_b.sum()
        for a in range(3):
            n_ab = N3[a, b, :]
            if n_ab.sum() == 0:
                continue
            p2 = n_ab / n_ab.sum()
            m = n_ab > 0
            G2 += 2 * float(np.sum(n_ab[m] * np.log(p2[m] / p1[m])))
        df += (int((N3[:, b, :].sum(axis=1) > 0).sum()) - 1) * (int((n_b > 0).sum()) - 1)
    pval = float(stats.chi2.sf(G2, df)) if df > 0 else np.nan
    # (五)表 4-7-3 實測時間占比兩種口徑
    tir = rt[["TBR", "TIR", "TAR"]].mean()
    pooled = pooled_counts / pooled_counts.sum() * 100
    summ = [
        ("記錄數", len(R)),
        ("兩步比對:一階模型 P² 之 30 分鐘出範圍機率(逐筆中位數,%)", float(R.model_short_risk_30m.median())),
        ("兩步比對:實際兩步出範圍比例(逐筆中位數,%)", float(R.empirical_out_of_range_30m.median())),
        ("兩步比對:逐筆差之中位數(實際−模型,百分點)",
         float((R.empirical_out_of_range_30m - R.model_short_risk_30m).median())),
        ("兩步比對:逐筆差絕對值之中位數(百分點)",
         float((R.empirical_out_of_range_30m - R.model_short_risk_30m).abs().median())),
        ("兩步比對:Spearman ρ", float(rho)),
        ("兩步比對:以各自中位數切分之高低分組一致筆數", int((mh == eh).sum())),
        ("程式核對:本檔重算之模型值與 patient_risk_table.csv 最大差(百分點,CSV 為 3 位小數)",
         float((R.model_short_risk_30m - R.csv_short_risk_30m).abs().max())),
        ("二階相依:目前範圍內且前一步範圍內 → 下一步出範圍(%)", p_prev_in),
        ("二階相依:目前範圍內且前一步低血糖 → 下一步出範圍(%)", p_prev_low),
        ("二階相依:目前範圍內且前一步高血糖 → 下一步出範圍(%)", p_prev_high),
        ("二階相依:樣本數(前一步範圍內/低/高)", f"{n_prev_in}/{n_prev_low}/{n_prev_high}"),
        ("概似比檢定 G²(二階 vs 一階,合併資料)", G2),
        ("概似比檢定自由度", df),
        ("概似比檢定 p 值", pval),
        ("表 4-7-3 實測占比(109 筆逐筆平均)TBR/TIR/TAR(%)",
         f"{tir.TBR:.4f}/{tir.TIR:.4f}/{tir.TAR:.4f}"),
        ("表 4-7-3 實測占比(全部讀值合併)TBR/TIR/TAR(%)",
         f"{pooled[0]:.4f}/{pooled[1]:.4f}/{pooled[2]:.4f}"),
    ]
    S = pd.DataFrame(summ, columns=["item", "value"])
    S.to_csv(f"{OUT}/markov_order_summary.csv", index=False, encoding="utf-8-sig")
    for k, v in summ:
        print(f"  {k}: {v}")
    return R, S


def main():
    print("=== (一)(二) 特徵集與對數轉換之敏感度(LR,100 次重複)===")
    feature_transform_sensitivity()
    print("\n=== (三) 成本參數敏感度 ===")
    cost_param_sensitivity()
    print("\n=== (四)(五) 一階馬可夫假設之檢查 ===")
    markov_order_check()


if __name__ == "__main__":
    main()
