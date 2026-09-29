# -*- coding: utf-8 -*-
"""
verify_outputs.py —— 第 1 層:本次輸出與「凍結快照」(reference/)逐格比對
================================================================
本層回答的問題只有一個:「另一次完整重跑所得之輸出,與先前凍結之那一次
執行是否相同?」因此必須是「兩次不同的執行」互相比對。若快照後沒有重跑,
比對的就是同一批檔案與其複本(稽核 B-01:封包之 38/38 PASS 即屬此類),
本程式會判為「自我比對(未重跑)」而失敗。

────────────────────────────────────────────────────────────────
正確流程(學生在 Windows 上):
  1. python run_all.py --clean --with-sensitivity     ← 第一次完整執行
  2. python verify_outputs.py --make-reference         ← 凍結快照(已有快照時會拒絕,
                                                          確定要換才加 --force)
  3. python run_all.py --clean --with-sensitivity     ← 第二次完整執行(獨立重跑)
  4. python verify_outputs.py                          ← 同平台比對,應全數通過
  另一台不同作業系統之機器重跑時,改用 --cross-platform(見下)。
  不可在第 3 步之前做第 4 步,也不可「先 --make-reference 再驗證」——那只是自我比對。
────────────────────────────────────────────────────────────────

〔稽核修正 B-01〕快照出處與自我比對防呆
  (a) --make-reference 寫入 reference/_reference_meta.json(format 2):建立時間、
      主機名稱、作業系統、Python 與套件版本、當時 run_log.json 之 SHA-256 與其
      開始/結束時間與 argv、原始資料指紋、每個 CSV 之 SHA-256 與分級(清單＝manifest)。
      reference/ 已存在時拒絕覆寫;加 --force 才重建,舊快照移至 reference_old_<時間>/。
      凍結前另檢查:run_log.json 必須存在、每支程式結束碼為 0、未用 --skip-lstm、
      且每個 CSV 之修改時間不早於該次執行之開始時間(否則代表有舊輸出殘留)。
  (b) 驗證時第一段即印出快照出處。以下任一成立即判「自我比對(未重跑)」,
      結束碼 3:①目前 run_log.json 與凍結時逐位元相同;②最近一次執行之開始時間
      早於快照建立時間;③全部輸出檔之修改時間都早於快照建立時間;④舊版快照
      (無出處資訊)且全部輸出與快照逐位元相同。
  (c) --allow-stale:仍做逐格比對,但訊息一律標示「形式檢查,非可重現性證據」。
  (d) 任一輸出檔之修改時間早於最近一次 run_all.py 之開始時間 → NOT-REGENERATED
      (該檔不是最近一次執行產生的,拿它比對不構成證據;例如 pooled_ph.py 續跑
      略過、或未加 --with-sensitivity)。

〔稽核修正 B-02〕判定邏輯
  SHAPE-DIFFER、READ-ERROR、NEW-FILE、TEXT-DIFFER、MISSING-FILE、COLUMNS-DIFFER、
  NOT-REGENERATED 一律計入失敗;NaN 位置不一致一律為失敗(不再被 np.nanmax 吞掉);
  同平台模式下 cv(XGBoost 相依)與 deep(LSTM 相依)級之差異同樣使結束碼為 1。
  另實作原檔頭所稱「勝過基準的病患數」檢查:凡同時有 lstm_rmse 與 baseline_rmse
  之檔(lstm_pooled_vs_baseline、lstm_vs_baseline_all、lstm_pooled_PH30m/60m),
  計數 lstm_rmse < baseline_rmse 之筆數;同平台須相同,跨平台列出差異。

〔稽核修正 B-11〕分級與容差(依據為下列兩份實測;細節見稽核修正筆記)
  級別          範圍                                     容差
  ─────────── ─────────────────────────────── ─────────────────────
  deterministic Markov、CGM 指標、持續性基準、EDA、LR 係數  |a−b| ≤ 1e-9 + 1e-9·|b|
  (確定性)    與勝算比、分群、生命表、LR/線性迴歸之效能列、  (絕對＋相對混合)
                OOF 風險表與象限、蒙地卡羅成本(固定種子)、    同平台與跨平台皆嚴格
                LSTM 檔內之基準欄(baseline_*、n_*、fold…)、
                診斷檔之非 LSTM 項、共線性 3 檔(collinearity_*)
  cv            XGBoost 相依:model_results/reg_results 之      同平台 1e-6 + 1e-6·|b|
  (XGBoost 相依)XGB 列、importance_*、reg_importance、epv 與   跨平台 → 只列警告
                敏感度表之 XGB 列
  deep          LSTM 相依欄:lstm_rmse、lstm_mae、skill_pct、    同平台:相對 5%(分母取 max(|b|,
  (LSTM 相依)  時界彙總之 LSTM 欄、依長度分組、診斷檔之        該檔 LSTM 相依各格 |b| 之中位數))
                LSTM 衍生 14 項                                 ＋勝過基準人數須相同
                                                                跨平台 → 只列警告

  實測一(封包輸出 vs 封包 reference/):38/38 逐位元相同——但 reference/ 是輸出之
    複本(B-01),此結果只證明兩者是同一批檔,不能校準容差;本版改判「自我比對」。
  實測二(稽核之 Linux 獨立完整重跑 vs 封包 reference/〔Windows〕,--cross-platform):
    ・確定性:13 檔逐位元相同(成本 4、馬可夫 6、cgm_metrics、cgm_descriptive、
      baseline_all)、5 檔僅換行符不同(數值相同)、lr_oddsratio 最大差 2.22e-15、
      model_results 之 LR 列差 0、reg_results 之 LR 列最大差 2.13e-14;
      GMM BIC(量級 3,285)差 1.575e-9——舊版以絕對 1e-9 判為「確定性不符」之誤報,
      混合容差下之門檻為 3.3e-6,通過(相對差僅 4.8e-13);而任何實質改動(例如
      成本 ×1.10、AUC ＋0.05、第 4 位小數之改變)皆遠大於 1e-9·|b|,必然失敗。
    ・XGBoost 相依:跨作業系統 model_results 之 XGB 列最大差 0.097(Precision_rep_min;
      AUC 中位數差至 0.0013)、迴歸 R² 差 0.019、重要度排序改變;同一作業系統重跑則位元
      相同(稽核 Linux 重跑與 reference_teacher 7 檔逐位元相同)→ 同平台嚴格、跨平台無法
      以容差吸收,只列警告。
    ・實測二之結果:同平台模式 exit 1(XGB/LSTM 相依 14 檔 FAIL、missing_pct 1 檔
      TEXT-DIFFER——皆屬預期);--cross-platform 模式 exit 0(24 檔 PASS、14 檔 WARN;
      WARN 檔中之確定性欄亦全數相符)。修改前之版本對同一組資料 exit 1,且唯一之「確定性不符」為 GMM BIC 誤報。
    ・LSTM 相依:跨作業系統逐筆 lstm_rmse 相對差至 22%、skill_pct 差至 22.9 個
      百分點;勝過基準人數 合併 107→108、PH30 100→98、PH60 91→90、逐人 72→72。
    ・missing_pct.csv:數值相同、同分之排列順序不同(排序未指定穩定演算法)→
      跨平台模式以「依 feature 排序後比對」處理;同平台仍須順序相同。

  --cross-platform 只放寬 cv/deep 級之「數值差異」;欄位、列數、缺檔、多檔、NaN
  位置、確定性內容一律維持嚴格。

用法:
    python verify_outputs.py                        # 同平台比對(預設)
    python verify_outputs.py --cross-platform       # 跨作業系統比對
    python verify_outputs.py --make-reference       # 凍結快照(已有快照時拒絕)
    python verify_outputs.py --make-reference --force
    python verify_outputs.py --allow-stale          # 形式檢查(非可重現性證據)
    python verify_outputs.py --stage-dir <目錄>     # 指定「程式與輸出」目錄(預設＝本檔所在)
    python verify_outputs.py --expect 41            # 另要求恰好逐格比對 N 個檔
結束碼:0 通過;1 未通過;2 無從比對(無快照/無輸出/拒絕凍結);3 自我比對(未重跑)

[修正 24] 之「空跑即通過」修補(以 reference/ 為權威清單、無輸出即失敗)仍保留。
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
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIRS = ["output", "output/eda_figures", "output_reg",
           "output_risk", "output_cost_mc", "cgm_output"]
META = "_reference_meta.json"
META_FORMAT = 2

# 〔稽核修正 B-01〕快照出處要記錄之套件(以發行套件名查版本,不必 import)
PACKAGES = ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "seaborn",
            "openpyxl", "xlrd", "xgboost", "shap", "tensorflow", "keras"]

# 只有加 --with-sensitivity 才會產生之輸出(缺檔時提示用)
SENSITIVITY_ONLY = ("output/sensitivity_missing_features.csv", "output/epv_vs_performance.csv",
                    "output/diagnostics_v2.csv", "output/lstm_by_length_group.csv",
                    "output/collinearity_",
                    # 〔稽核修正 A-14/A-15/A-21/A-26〕sensitivity_extra.py 之 5 個輸出
                    "output/sensitivity_features_transform.csv", "output/sensitivity_skewness_table.csv",
                    "output/sensitivity_cost_params.csv", "output/markov_order_")

# ---------------------------------------------------------------------------
# 〔稽核修正 B-11〕分級規則
#   分級碼:0＝deterministic(確定性)、1＝cv(XGBoost 相依)、2＝deep(LSTM 相依)
#   RULES:檔案 → [(列條件, 欄集合, 分級), ...];由上而下,每一格取第一條命中者;
#          未命中之格一律為確定性。
#     列條件:None＝全部列;("欄名", 正規式)＝以「參考快照」該欄之值判定之列。
#     欄集合:None＝全部欄(含文字欄);否則為欄名集合(只作用於列出之欄)。
# ---------------------------------------------------------------------------
DET, CV, DEEP = 0, 1, 2
GNAME = {DET: "deterministic", CV: "cv", DEEP: "deep"}
GZH = {DET: "確定性", CV: "XGB相依", DEEP: "LSTM相依"}
TOL = {DET: (1e-9, 1e-9),       # (絕對, 相對)
       CV: (1e-6, 1e-6),
       DEEP: (0.0, 0.05)}       # 相對 5%;分母見 _tol_vector()

_LSTM_COLS = {"lstm_rmse", "lstm_mae", "skill_pct"}
# 診斷檔中由 LSTM 結果衍生之 14 項:稽核 Linux 重跑中與封包不同者 12 項(skill、gap、
#   偏相關等),另 beatpct_* 2 項為逐人勝率,該次恰巧相同,但本質上依賴 LSTM。
DIAG_DEEP_RE = (r"^(?:lstm_(?:per|pooled)_skill_|lstm_gap_|corr_comp_skill|partial_comp_skill"
                r"|skill_median_|beatpct_)")
RULES = {
    "output/model_results.csv": [(("Model", r"^XGB"), None, CV)],
    "output_reg/reg_results.csv": [(("Model", r"^XGB"), None, CV)],
    "output/epv_vs_performance.csv": [
        (("model", r"^XGB"), {"AUC_median", "AUC_lo95", "AUC_hi95", "CI_width"}, CV)],
    "output/sensitivity_missing_features.csv": [
        (("model", r"^XGB"), {"AUC_median", "AUC_lo95", "AUC_hi95"}, CV)],
    "output/importance_Any_Complication.csv": [(None, None, CV)],
    "output/importance_Macrovascular.csv": [(None, None, CV)],
    "output/importance_Microvascular.csv": [(None, None, CV)],
    "output_reg/reg_importance_HbA1c.csv": [(None, None, CV)],
    "cgm_output/lstm_pooled_PH30m.csv": [(None, _LSTM_COLS, DEEP)],
    "cgm_output/lstm_pooled_PH60m.csv": [(None, _LSTM_COLS, DEEP)],
    "cgm_output/lstm_pooled_vs_baseline.csv": [(None, _LSTM_COLS, DEEP)],
    "cgm_output/lstm_vs_baseline_all.csv": [(None, _LSTM_COLS, DEEP)],
    "cgm_output/lstm_pooled_by_horizon.csv": [
        (None, {"lstm_rmse_mean", "lstm_rmse_median", "lstm_mae_mean", "skill_median",
                "beat_pct", "pooled_rmse", "r2"}, DEEP)],
    "output/lstm_by_length_group.csv": [
        (None, {"per_skill_median", "pooled_skill_median", "per_beat_pct",
                "pooled_beat_pct"}, DEEP)],
    "output/diagnostics_v2.csv": [(("item", DIAG_DEEP_RE), {"value"}, DEEP)],
}
# 全檔確定性者(明列,以便報告「已分級」;未列亦未在 RULES 者依檔名推定並加註)
DETERMINISTIC_FILES = {
    "cgm_output/markov_3state.csv", "cgm_output/markov_5state.csv",
    "cgm_output/markov_convergence_3state.csv", "cgm_output/markov_convergence_5state.csv",
    "cgm_output/markov_stationary_3state.csv", "cgm_output/markov_stationary_5state.csv",
    "cgm_output/cgm_metrics.csv", "cgm_output/cgm_descriptive.csv",
    "cgm_output/baseline_all.csv",
    "output/cluster_complication_rate.csv", "output/cluster_selection.csv",
    "output/eda_figures/correlation_matrix.csv", "output/eda_figures/describe.csv",
    "output/eda_figures/missing_pct.csv",
    "output/lr_oddsratio_Any_Complication.csv", "output/lr_oddsratio_Macrovascular.csv",
    "output/lr_oddsratio_Microvascular.csv",
    "output_cost_mc/life_table_ex.csv", "output_cost_mc/patient_lifetime_cost_mc.csv",
    "output_cost_mc/quadrant_cost_interval.csv",
    "output_cost_mc/cost_sensitivity_class_weight.csv",
    "output_risk/patient_risk_table.csv", "output_risk/quadrant_summary.csv",
    # 〔稽核修正 B-10 介面〕CODE 修正員新增之共線性覆核輸出(僅 --with-sensitivity)
    "output/collinearity_vif.csv", "output/collinearity_bootstrap_signs.csv",
    "output/collinearity_ablation_auc.csv",
    # 〔稽核修正 A-14/A-15/A-21/A-26〕sensitivity_extra.py 之輸出(僅 --with-sensitivity;LR、
    #   固定種子之蒙地卡羅與 CGM 計數,皆為確定性)
    "output/sensitivity_features_transform.csv", "output/sensitivity_skewness_table.csv",
    "output/sensitivity_cost_params.csv", "output/markov_order_check.csv",
    "output/markov_order_summary.csv",
}
# 跨平台模式下「依鍵排序後比對」之檔(同分排序不穩定;稽核 CODE-23)
KEY_SORT_XP = {"output/eda_figures/missing_pct.csv": ["feature"]}
# 勝過基準人數:同時具有此二欄之檔
BEAT_COLS = ("lstm_rmse", "baseline_rmse")


# ============================================================ 小工具
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_files(stage):
    out = []
    for d in OUTDIRS:
        p = os.path.join(stage, d)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.endswith(".csv"):
                out.append(f"{d}/{f}")
    return out


def ref_files(ref):
    out = []
    for root, _dirs, fns in os.walk(ref):
        for fn in sorted(fns):
            if fn.endswith(".csv"):
                out.append(os.path.relpath(os.path.join(root, fn), ref).replace(os.sep, "/"))
    return sorted(out)


def _txt(series):
    """文字欄轉為可比較之字串陣列;缺值一律為 "<NA>"(pandas 3 之字串型別會保留 NaN,nan != nan)。"""
    return series.astype(object).where(series.notna(), "<NA>").astype(str).to_numpy()


def _fmt_time(epoch):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


def _parse_local(s):
    """run_log.json 之 started/finished 為本機時間字串(無時區)。"""
    try:
        return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))
    except Exception:
        return None


def read_runlog(stage):
    p = os.path.join(stage, "run_log.json")
    if not os.path.exists(p):
        return None
    try:
        j = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        return {"_error": str(e), "_path": p}
    steps = j.get("steps", []) or []
    return {
        "_path": p,
        "sha256": sha256_file(p),
        "mtime": os.path.getmtime(p),
        "started": j.get("started"),
        "finished": j.get("finished"),
        "runner_argv": j.get("runner_argv"),
        "with_sensitivity": j.get("with_sensitivity"),
        "skip_lstm": j.get("skip_lstm"),
        "n_steps": len(steps),
        "failed": [f"{s.get('script')}(rc={s.get('returncode')})" for s in steps
                   if s.get("returncode") != 0],
        "scripts": [s.get("script") for s in steps],
    }


def load_meta(ref):
    p = os.path.join(ref, META)
    if not os.path.exists(p):
        return {"_legacy": True, "_missing": True}
    try:
        m = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        return {"_legacy": True, "_error": str(e)}
    if m.get("format") != META_FORMAT:
        m["_legacy"] = True
    return m


def pkg_versions():
    out = {}
    try:
        from importlib import metadata as _md
    except Exception:                                   # pragma: no cover
        _md = None
    for p in PACKAGES:
        try:
            out[p] = _md.version(p) if _md else "unknown"
        except Exception:
            out[p] = "未安裝"
    return out


# ============================================================ 分級
def build_grades(rel, B, num_cols, txt_cols):
    """回傳 (數值欄 → 每列分級陣列, 文字欄 → 分級, 分級說明, 是否已明列)。"""
    n = len(B)
    g_num = {c: np.full(n, DET, dtype=int) for c in num_cols}
    g_txt = {c: DET for c in txt_cols}
    rules = RULES.get(rel)
    if rules is None:
        if rel in DETERMINISTIC_FILES:
            return g_num, g_txt, "確定性", True
        if os.path.basename(rel).startswith("lstm_"):
            for c in num_cols:
                if not (c.startswith("baseline") or c.startswith("n_") or c in ("patient", "fold", "PH_min")):
                    g_num[c][:] = DEEP
            return g_num, g_txt, "未分級→依檔名推定 LSTM 相依", False
        return g_num, g_txt, "未分級→確定性(預設)", False
    done = {c: np.zeros(n, dtype=bool) for c in num_cols}
    for cond, cols, g in rules:
        if cond is None:
            rowmask = np.ones(n, dtype=bool)
        else:
            col, rx = cond
            if col not in B.columns:
                continue
            rowmask = B[col].astype(object).where(B[col].notna(), "").astype(str) \
                .str.contains(rx, regex=True, na=False).to_numpy(dtype=bool)
        for c in num_cols:
            if cols is None or c in cols:
                m = rowmask & ~done[c]
                g_num[c][m] = g
                done[c] |= m
        if cond is None:
            for c in txt_cols:
                if cols is None or c in cols:
                    g_txt[c] = g
    present = set()
    for c in num_cols:
        present |= set(np.unique(g_num[c]).tolist())
    present |= set(g_txt.values())
    label = "＋".join(GZH[g] for g in sorted(present)) if present else "確定性"
    return g_num, g_txt, label, True


def _tol_vector(Y, g, m):
    """容差向量。deep 級:相對 5%,但分母取 max(|參考值|, 該級各格 |參考值| 之中位數),
    避免技巧分數接近 0 之格因分母過小而失真。"""
    atol, rtol = TOL[g]
    if g == DEEP:
        ay = np.abs(Y)
        sel = ay[m & np.isfinite(ay)]
        scale = float(np.median(sel)) if sel.size else 0.0
        return rtol * np.maximum(np.where(np.isfinite(ay), ay, 0.0), scale)
    return atol + rtol * np.abs(Y)


# ============================================================ 逐檔比對
def compare(ref_p, cur_p, rel, xp):
    """回傳 dict(status, grade, detail, beat, notes)。
    status:PASS / WARN(僅跨平台、僅 cv 或 deep 差異)/ FAIL / TEXT-DIFFER /
            SHAPE-DIFFER / COLUMNS-DIFFER / READ-ERROR"""
    res = {"status": "PASS", "grade": "", "detail": "", "beat": None, "notes": []}
    try:
        A = pd.read_csv(cur_p)
        B = pd.read_csv(ref_p)
    except Exception as e:                              # 〔稽核修正 B-02〕計入失敗
        res.update(status="READ-ERROR", detail=str(e)[:80])
        return res
    if list(A.columns) != list(B.columns):
        extra = [c for c in A.columns if c not in B.columns]
        lack = [c for c in B.columns if c not in A.columns]
        res.update(status="COLUMNS-DIFFER",
                   detail=f"本次 {len(A.columns)} 欄/快照 {len(B.columns)} 欄;"
                          f"多 {extra[:4]} 少 {lack[:4]}" + ("(順序不同)" if not extra and not lack else ""))
        return res
    if A.shape != B.shape:                               # 〔稽核修正 B-02〕計入失敗
        res.update(status="SHAPE-DIFFER",
                   detail=f"本次 {A.shape[0]} 列/快照 {B.shape[0]} 列")
        return res

    # 〔稽核修正 B-11〕跨平台:同分排序不穩定之檔改依鍵排序後比對
    order_note = ""
    keys = KEY_SORT_XP.get(rel)
    if keys and all(k in A.columns for k in keys):
        same_order = all((_txt(A[k]) == _txt(B[k])).all() for k in keys)
        if not same_order:
            As = A.sort_values(keys, kind="mergesort").reset_index(drop=True)
            Bs = B.sort_values(keys, kind="mergesort").reset_index(drop=True)
            if xp:
                A, B = As, Bs
                order_note = "列順序不同(同分排序);已依 " + "/".join(keys) + " 排序後比對"
            else:
                sorted_same = As.equals(Bs)
                order_note = ("列順序不同;依 " + "/".join(keys) + " 排序後" +
                              ("逐格相同(僅同分排序不同——跨作業系統請用 --cross-platform)"
                               if sorted_same else "仍不同"))

    num = [c for c in A.columns
           if pd.api.types.is_numeric_dtype(A[c]) and pd.api.types.is_numeric_dtype(B[c])]
    txt = [c for c in A.columns if c not in num]
    g_num, g_txt, glabel, _known = build_grades(rel, B, num, txt)
    res["grade"] = glabel

    stats = {g: {"cells": 0, "bad": 0, "maxabs": 0.0, "maxrel": 0.0, "worst": None,
                 "nan": 0, "nan_ex": None, "txt": 0, "txt_ex": None} for g in (DET, CV, DEEP)}
    for c in num:
        X = A[c].to_numpy(dtype=float)
        Y = B[c].to_numpy(dtype=float)
        nx, ny = np.isnan(X), np.isnan(Y)
        with np.errstate(all="ignore"):
            d = np.abs(X - Y)
        d[(nx & ny) | (X == Y)] = 0.0                   # 同為 NaN 或完全相等(含同號 inf)
        nan_bad = nx != ny                               # 〔稽核修正 B-02〕NaN 位置不一致
        gv = g_num[c]
        for g in (DET, CV, DEEP):
            m = gv == g
            if not m.any():
                continue
            st = stats[g]
            st["cells"] += int(m.sum())
            nb = nan_bad & m
            if nb.any():
                st["nan"] += int(nb.sum())
                if st["nan_ex"] is None:
                    i = int(np.where(nb)[0][0])
                    st["nan_ex"] = (i, c, Y[i], X[i])
            ok_m = m & ~nan_bad
            if not ok_m.any():
                continue
            tol = _tol_vector(Y, g, m)
            dd = np.where(ok_m, d, 0.0)
            dd = np.where(np.isnan(dd), np.inf, dd)
            bad = ok_m & (dd > tol)
            st["bad"] += int(bad.sum())
            if dd.max() > st["maxabs"]:
                i = int(np.argmax(dd))
                st["maxabs"] = float(dd[i])
                st["worst"] = (i, c, Y[i], X[i])
            with np.errstate(all="ignore"):
                rel_ = np.where(ok_m & (np.abs(Y) > 1e-300), dd / np.abs(Y), 0.0)
            rel_ = rel_[np.isfinite(rel_)]
            if rel_.size and rel_.max() > st["maxrel"]:
                st["maxrel"] = float(rel_.max())
    for c in txt:
        a = _txt(A[c])
        b = _txt(B[c])
        ne = a != b
        g = g_txt[c]
        stats[g]["cells"] += len(a)
        if ne.any():
            stats[g]["txt"] += int(ne.sum())
            if stats[g]["txt_ex"] is None:
                i = int(np.where(ne)[0][0])
                stats[g]["txt_ex"] = (i, c, b[i], a[i])

    hard, soft, text_only_hard = [], [], True
    parts = []
    for g in (DET, CV, DEEP):
        st = stats[g]
        if not st["cells"]:
            continue
        tol_s = {DET: "1e-9＋1e-9·|參考|", CV: "1e-6＋1e-6·|參考|", DEEP: "相對 5%"}[g]
        issues = []
        if st["bad"]:
            i, c, y, x = st["worst"]
            issues.append(f"{st['bad']} 格超出容差({tol_s}),最大差 {st['maxabs']:.4g}"
                          f"(第 {i+1} 列「{c}」:快照 {y:.10g}、本次 {x:.10g})")
        if st["nan"]:
            i, c, y, x = st["nan_ex"]
            issues.append(f"NaN 位置不一致 {st['nan']} 格(第 {i+1} 列「{c}」:快照 {y:.10g}、本次 {x:.10g})")
        if st["txt"]:
            i, c, y, x = st["txt_ex"]
            issues.append(f"文字不同 {st['txt']} 格(第 {i+1} 列「{c}」:快照 {y!r}、本次 {x!r})")
        if issues:
            is_soft = xp and g in (CV, DEEP)
            (soft if is_soft else hard).append(f"[{GZH[g]}] " + ";".join(issues))
            if not is_soft and (st["bad"] or st["nan"]):
                text_only_hard = False
        else:
            parts.append(f"{GZH[g]} {st['cells']} 格 最大差 {st['maxabs']:.3g}")

    # 〔稽核修正 B-02〕勝過基準的病患數
    if all(k in A.columns for k in BEAT_COLS):
        rb = int((B[BEAT_COLS[0]] < B[BEAT_COLS[1]]).sum())
        ra = int((A[BEAT_COLS[0]] < A[BEAT_COLS[1]]).sum())
        res["beat"] = (rb, ra, len(A))
        if rb != ra:
            msg = f"[勝過基準人數] 快照 {rb}/{len(B)} → 本次 {ra}/{len(A)}"
            if xp:
                soft.append(msg)
            else:
                hard.append(msg)
                text_only_hard = False

    if order_note:
        res["notes"].append(order_note)
    if hard:
        res["status"] = "TEXT-DIFFER" if text_only_hard else "FAIL"
        res["detail"] = ";".join(hard + soft)
    elif soft:
        res["status"] = "WARN"
        res["detail"] = ";".join(soft)
    else:
        res["detail"] = ";".join(parts) if parts else "無數值欄"
    if order_note:
        res["detail"] = order_note + ";" + res["detail"]
    return res


# ============================================================ 快照出處
def provenance_lines(meta, ref):
    L = []
    if meta.get("_legacy"):
        if meta.get("_missing"):
            L.append("[快照出處] ★ reference/ 內沒有 _reference_meta.json —— 快照出處不明。")
        else:
            L.append("[快照出處] ★ 舊版快照(_reference_meta.json 只有檔數,無建立時間、主機、"
                     "執行紀錄)—— 無法證明它凍結的是哪一次執行。")
        rp = os.path.join(ref, "data_manifest.json")
        if os.path.exists(rp):
            L.append(f"           reference/ 資料夾之修改時間:{_fmt_time(os.path.getmtime(ref))}"
                     "(僅供參考;複製或解壓縮會改變)")
        L.append("           建議:依檔頭「正確流程」完整重跑後以 --make-reference --force 重新凍結。")
        return L
    rl = meta.get("run_log") or {}
    L.append(f"[快照出處] 建立於 {meta.get('created')}(主機 {meta.get('hostname')};"
             f"{meta.get('os')};Python {meta.get('python')})")
    L.append(f"           凍結指令:{' '.join(meta.get('argv') or [])}")
    if rl:
        L.append(f"           凍結當時之 run_log.json:開始 {rl.get('started')}、結束 {rl.get('finished')}、"
                 f"argv {' '.join(rl.get('runner_argv') or []) or '(未記錄)'}、"
                 f"SHA-256 {str(rl.get('sha256'))[:16]}…")
    else:
        L.append("           ★ 凍結當時沒有 run_log.json(無法得知快照來自哪一次執行)")
    pk = meta.get("packages") or {}
    if pk:
        L.append("           套件:" + "、".join(f"{k} {v}" for k, v in pk.items()))
    L.append(f"           快照清單(manifest):{meta.get('n_files')} 個 CSV;原始資料指紋 "
             f"{str(meta.get('data_manifest_sha'))[:16]}…")
    if meta.get("forced_warnings"):
        L.append("           ★ 此快照以 --force 建立,當時之警告:")
        for w in meta["forced_warnings"][:8]:
            L.append(f"             - {w}")
    return L


# ============================================================ 凍結快照
def make_reference(stage, force):
    ref = os.path.join(stage, "reference")
    print("【第 1 層】凍結快照(--make-reference)")
    if os.path.isdir(ref) and not force:
        # 〔稽核修正 B-01〕不得以本次輸出覆寫既有快照(那會把驗證變成自我比對)
        meta = load_meta(ref)
        print("[拒絕] reference/ 已存在,不覆寫。")
        for l in provenance_lines(meta, ref):
            print(l)
        print("  快照是「某一次完整執行」之凍結結果;驗證時要拿「另一次完整重跑」與它比對。")
        print("  若確定要換一個快照(例如程式依稽核修正改變了輸出,且剛以")
        print("  run_all.py --clean --with-sensitivity 完整重跑過),請改用:")
        print("      python verify_outputs.py --make-reference --force")
        print("  舊快照會被移到 reference_old_<時間>/ 保存。")
        sys.exit(2)
    cur = csv_files(stage)
    if not cur:
        print("[錯誤] 找不到任何輸出 CSV,無可凍結。請先執行 run_all.py --clean --with-sensitivity。")
        sys.exit(2)
    rl = read_runlog(stage)
    problems = []
    if rl is None:
        problems.append("找不到 run_log.json —— 無法證明這些輸出來自哪一次執行")
    elif "_error" in rl:
        problems.append(f"run_log.json 無法讀取:{rl['_error']}")
    else:
        if rl["failed"]:
            problems.append("最近一次 run_all.py 有程式失敗:" + "、".join(rl["failed"]))
        if rl.get("skip_lstm"):
            problems.append("最近一次 run_all.py 使用 --skip-lstm(LSTM 結果未重新產生)")
        st = _parse_local(rl.get("started") or "")
        if st is not None:
            old = [r for r in cur if os.path.getmtime(os.path.join(stage, r)) < st - 2]
            if old:
                problems.append(f"{len(old)} 個 CSV 早於最近一次執行之開始時間 {rl.get('started')}"
                                f"(不是該次執行產生的,例如:{', '.join(old[:4])})")
    if problems and not force:
        print("[拒絕] 目前的輸出不適合凍結為快照:")
        for p in problems:
            print(f"   ★ {p}")
        print("  請先執行:python run_all.py --clean --with-sensitivity,再凍結。")
        print("  (確定要照現況凍結者可加 --force,上述警告會寫入快照出處並在每次驗證時印出。)")
        sys.exit(2)
    if os.path.isdir(ref):
        dst = os.path.join(stage, "reference_old_" + time.strftime("%Y%m%d_%H%M%S"))
        shutil.move(ref, dst)
        print(f"[reference] 舊快照已移至 {os.path.basename(dst)}/(確認不需要後可自行刪除)")
    files = {}
    for rel in cur:
        src = os.path.join(stage, rel)
        dst = os.path.join(ref, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        try:
            B = pd.read_csv(src)
            num = [c for c in B.columns if pd.api.types.is_numeric_dtype(B[c])]
            txt = [c for c in B.columns if c not in num]
            _gn, _gt, glabel, _k = build_grades(rel, B, num, txt)
            shape = list(B.shape)
        except Exception as e:
            glabel, shape = f"(讀取失敗:{e})", None
        files[rel] = {"sha256": sha256_file(src), "bytes": os.path.getsize(src),
                      "shape": shape, "grade": glabel}
    for extra in ["data_manifest.json", "env_lock.json", "run_log.json"]:
        p = os.path.join(stage, extra)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(ref, extra))
    dm = None
    try:
        dm = json.load(open(os.path.join(stage, "data_manifest.json"), encoding="utf-8")).get("combined_sha256")
    except Exception:
        pass
    now = time.time()
    lt = time.localtime(now)
    off = -(time.altzone if (lt.tm_isdst > 0 and time.daylight) else time.timezone)   # Windows 之 %z 不可靠
    tz = f"{'+' if off >= 0 else '-'}{abs(off) // 3600:02d}{abs(off) % 3600 // 60:02d}"
    meta = {
        "format": META_FORMAT,
        "created": time.strftime("%Y-%m-%d %H:%M:%S", lt) + tz,
        "created_epoch": now,
        "hostname": platform.node(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "packages": pkg_versions(),
        "argv": [os.path.basename(sys.argv[0])] + sys.argv[1:],
        "run_log": None if (rl is None or "_error" in rl) else {
            k: rl[k] for k in ("sha256", "started", "finished", "runner_argv",
                               "with_sensitivity", "skip_lstm", "n_steps", "scripts")},
        "data_manifest_sha": dm,
        "n_files": len(files),
        "files": files,
        "forced_warnings": problems,
        "note": "快照＝某一次完整執行之凍結結果;驗證須以另一次完整重跑與之比對。",
    }
    with open(os.path.join(ref, META), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[reference] 已凍結 {len(files)} 個 CSV → reference/(含 {META})")
    for l in provenance_lines(meta, ref):
        print(l)
    print("下一步:再完整重跑一次(python run_all.py --clean --with-sensitivity),"
          "之後執行 python verify_outputs.py 比對。")
    print("★ 凍結之後立刻驗證,比對的是同一批檔案,程式會判為「自我比對(未重跑)」。")
    sys.exit(0)


# ============================================================ 主程式
def main():
    ap = argparse.ArgumentParser(description="第 1 層:本次輸出與凍結快照逐格比對")
    ap.add_argument("--make-reference", action="store_true", help="把目前輸出凍結為快照")
    ap.add_argument("--force", action="store_true", help="與 --make-reference 併用:覆寫既有快照")
    ap.add_argument("--cross-platform", action="store_true",
                    help="跨作業系統比對:XGBoost/LSTM 相依之數值差異只列警告")
    ap.add_argument("--allow-stale", action="store_true",
                    help="快照後未重跑仍執行比對(僅形式檢查,非可重現性證據)")
    ap.add_argument("--stage-dir", default="", help="「程式與輸出」目錄(預設為本檔所在目錄)")
    ap.add_argument("--expect", type=int, default=0,
                    help="宣告應逐格比對之 CSV 檔數;不符即失敗(預設以快照清單為準)")
    a = ap.parse_args()
    stage = os.path.abspath(a.stage_dir) if a.stage_dir else HERE
    ref = os.path.join(stage, "reference")

    if a.make_reference:
        make_reference(stage, a.force)
        return

    print("【第 1 層】本次輸出與凍結快照(reference/)逐格比對")
    print(f"[目錄] {stage}")
    if not os.path.isdir(ref):
        print("[錯誤] 找不到 reference/(凍結快照)。首次建立之流程:")
        print("   1. python run_all.py --clean --with-sensitivity")
        print("   2. python verify_outputs.py --make-reference")
        print("   3. python run_all.py --clean --with-sensitivity   (第二次、獨立重跑)")
        print("   4. python verify_outputs.py")
        sys.exit(2)

    meta = load_meta(ref)
    for l in provenance_lines(meta, ref):               # 〔稽核修正 B-01〕第一段即印出快照出處
        print(l)
    mode = ("跨平台(--cross-platform):XGBoost/LSTM 相依之數值差異只列警告;確定性內容與結構維持嚴格"
            if a.cross_platform else
            "同平台(預設):任何分級之差異皆為失敗;勝過基準人數須相同")
    print(f"[模式] {mode}")
    if a.allow_stale:
        print("[模式] ★ --allow-stale:本次只做「形式檢查」,結果不構成可重現性證據。")

    hard_msgs = []          # 使結束碼為 1 之全域問題
    # ---- 本次執行紀錄
    rl = read_runlog(stage)
    run_started = None
    if rl is None:
        print("[本次執行] ★ 找不到 run_log.json —— 無法確認輸出由哪一次執行產生。")
    elif "_error" in rl:
        print(f"[本次執行] ★ run_log.json 無法讀取:{rl['_error']}")
        hard_msgs.append("run_log.json 無法讀取")
    else:
        run_started = _parse_local(rl.get("started") or "")
        print(f"[本次執行] run_log.json:開始 {rl.get('started')}、結束 {rl.get('finished')}、"
              f"argv {' '.join(rl.get('runner_argv') or []) or '(未記錄)'}、{rl['n_steps']} 支程式")
        if rl["failed"]:
            print("[本次執行] ★ 有程式失敗:" + "、".join(rl["failed"]))
            hard_msgs.append("最近一次 run_all.py 有程式失敗")
        if rl.get("skip_lstm"):
            print("[本次執行] ★ 使用了 --skip-lstm,LSTM 結果未重新產生")
            hard_msgs.append("最近一次 run_all.py 使用 --skip-lstm")

    # ---- 原始資料指紋(〔稽核修正 B-02〕不一致改為失敗;原版只印字)
    rp, cp = os.path.join(ref, "data_manifest.json"), os.path.join(stage, "data_manifest.json")
    if os.path.exists(rp) and os.path.exists(cp):
        r = json.load(open(rp, encoding="utf-8"))
        c = json.load(open(cp, encoding="utf-8"))
        same = r.get("combined_sha256") == c.get("combined_sha256")
        print(f"[資料] 原始資料指紋 {'一致' if same else '★不一致★ — 資料已被改動'}")
        if not same:
            hard_msgs.append("原始資料指紋與快照不一致")
    else:
        print("[資料] ★ 缺 data_manifest.json(快照或本次),無法比對原始資料指紋")

    cur_list = csv_files(stage)
    ref_list = ref_files(ref)
    # ---- 快照完整性(format 2:與凍結時之 manifest 逐檔核對)
    if not meta.get("_legacy"):
        mf = meta.get("files") or {}
        bad_ref = []
        for rel in sorted(set(mf) | set(ref_list)):
            if rel not in mf:
                bad_ref.append(f"{rel}(快照內多出、未登錄)")
            elif rel not in ref_list:
                bad_ref.append(f"{rel}(已登錄、快照內缺)")
            elif sha256_file(os.path.join(ref, rel)) != mf[rel].get("sha256"):
                bad_ref.append(f"{rel}(內容與凍結時不同)")
        if bad_ref:
            print(f"[快照] ★ 快照在凍結後被改動({len(bad_ref)} 項):" + ";".join(bad_ref[:5]))
            hard_msgs.append("快照在凍結後被改動")
        else:
            print(f"[快照] 快照內 {len(ref_list)} 個 CSV 與凍結時之 SHA-256 逐檔相符")

    # ---- [修正 24b] 什麼都沒比對,不能算通過
    if not cur_list and not ref_list:
        print("[錯誤] 本次輸出與 reference/ 皆無任何 CSV,無從比對。"
              "請確認 --stage-dir 或工作目錄(須含 output*/、cgm_output/)。")
        sys.exit(2)
    if not cur_list:
        print(f"[★★★] 本次輸出目錄中找不到任何 CSV,但 reference/ 有 {len(ref_list)} 個。")
        print("       可能原因:(1) 尚未執行 run_all.py;(2) 目錄不對;(3) 輸出目錄被刪除或改名。")
        print("       ——「沒有東西可比」不是「全部通過」,故以失敗結束。")
        sys.exit(2)

    # ---- 〔稽核修正 B-01〕自我比對(未重跑)判定
    stale = []
    newest = max(os.path.getmtime(os.path.join(stage, r)) for r in cur_list)
    if not meta.get("_legacy"):
        mrl = meta.get("run_log") or {}
        if rl and "_error" not in rl and mrl.get("sha256") and rl["sha256"] == mrl["sha256"]:
            stale.append("目前的 run_log.json 與凍結快照時逐位元相同(快照後沒有執行過 run_all.py)")
        if run_started is not None and meta.get("created_epoch") and run_started < meta["created_epoch"]:
            stale.append(f"最近一次 run_all.py 開始於 {rl.get('started')},早於快照建立時間 {meta.get('created')}")
        if meta.get("created_epoch") and newest < meta["created_epoch"]:
            stale.append(f"全部 {len(cur_list)} 個輸出檔之修改時間(最新 {_fmt_time(newest)})"
                         f"都早於快照建立時間 {meta.get('created')}")
    else:
        same_all = (set(cur_list) == set(ref_list)) and all(
            sha256_file(os.path.join(stage, r)) == sha256_file(os.path.join(ref, r)) for r in ref_list)
        if same_all:
            stale.append(f"舊版快照(出處不明),且全部 {len(ref_list)} 個輸出與快照逐位元相同"
                         "——這是檔案與其複本之比對")
    if stale:
        print("[自我比對] ★ 判定為「自我比對(快照後未重跑)」:")
        for s in stale:
            print(f"   - {s}")
        if not a.allow_stale:
            print("   這種比對必然全數相同,不構成任何可重現性證據(稽核 B-01)。正確流程:")
            print("     python run_all.py --clean --with-sensitivity   ← 先完整重跑")
            print("     python verify_outputs.py                       ← 再比對")
            print("   (只想檢查檔案格式者可加 --allow-stale,結果會標示為形式檢查。)")
            print("\n結論:未通過 —— 自我比對(未重跑)。")
            sys.exit(3)
        print("   已指定 --allow-stale:以下比對僅為形式檢查,★ 非可重現性證據 ★")
    elif meta.get("_legacy"):
        print("[自我比對] 舊版快照:輸出與快照並非逐位元相同(但因快照無出處資訊,"
              "無法確認差異是否來自另一次完整執行)。")
    else:
        print("[自我比對] 未發現自我比對跡象:run_log.json 已更新、且有輸出檔晚於快照建立時間。")

    missing = [r for r in ref_list if r not in set(cur_list)]
    print(f"[範圍] 快照清單 {len(ref_list)} 個 CSV;本次輸出 {len(cur_list)} 個;"
          f"快照有而本次缺 {len(missing)} 個;本次多出 {len(set(cur_list) - set(ref_list))} 個")

    rows, beats = [], []
    for rel in cur_list:
        ref_p = os.path.join(ref, rel)
        cur_p = os.path.join(stage, rel)
        if not os.path.exists(ref_p):
            rows.append((rel, "—", "NEW-FILE",
                         "快照中沒有這個檔(若為程式新增之輸出,須完整重跑後重新凍結快照)"))
            continue
        r = compare(ref_p, cur_p, rel, a.cross_platform)
        st, det = r["status"], r["detail"]
        # 〔稽核修正 B-01(d)〕不是最近一次執行產生的檔
        if run_started is not None and os.path.getmtime(cur_p) < run_started - 2 and not a.allow_stale:
            det = (f"修改時間 {_fmt_time(os.path.getmtime(cur_p))} 早於最近一次執行開始 "
                   f"{rl.get('started')}(非該次產生);原比對結果 {st}:{det}")
            st = "NOT-REGENERATED"
        rows.append((rel, r["grade"], st, det))
        if r["beat"] is not None:
            beats.append((rel, r["beat"]))
    for rel in missing:
        hint = ("(最近一次執行可能未加 --with-sensitivity)"
                if rel.startswith(SENSITIVITY_ONLY) else "")
        rows.append((rel, "—", "MISSING-FILE", "快照有、本次未產生" + hint))

    R = pd.DataFrame(rows, columns=["檔案", "分級", "結果", "說明"])
    R = R.sort_values("檔案").reset_index(drop=True)
    pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_colwidth", 160)
    print(R.to_string(index=False))

    if beats:
        print("\n[勝過基準的病患數](lstm_rmse < baseline_rmse 之筆數;同平台須相同)")
        for rel, (rb, ra, n) in beats:
            flag = "相同" if rb == ra else ("★不同(跨平台:列為警告)" if a.cross_platform else "★不同")
            print(f"   {rel:<42} 快照 {rb}/{n}  本次 {ra}/{n}  {flag}")

    counts = R["結果"].value_counts()
    n_pass = int(counts.get("PASS", 0))
    n_warn = int(counts.get("WARN", 0))
    n_bad = int(len(R) - n_pass - n_warn)
    n_cmp = int(R["結果"].isin(["PASS", "WARN", "FAIL", "TEXT-DIFFER"]).sum())
    n_cmp_bad = int(R["結果"].isin(["FAIL", "TEXT-DIFFER"]).sum())
    print("\n" + "=" * 72)
    print(counts.to_string())
    print(f"\n共 {len(R)} 個檔:PASS {n_pass}、WARN {n_warn}、失敗 {n_bad}"
          f"(逐格比對後不符 {n_cmp_bad}、未能逐格比對/缺檔/多檔/非最近一次產生 {n_bad - n_cmp_bad});"
          f"實際逐格比對 {n_cmp} 個檔")

    bad_expect = False
    if a.expect and n_cmp != a.expect:                  # [修正 24e]
        bad_expect = True
        print(f"★ 宣告應逐格比對 {a.expect} 個檔,實際 {n_cmp} 個 —— 覆蓋範圍不符。")
    for label, why in [("MISSING-FILE", "快照有而本次未產生 —— 重現不完整"),
                       ("NEW-FILE", "本次多出快照所無之檔 —— 程式輸出已改變,須重新凍結快照"),
                       ("COLUMNS-DIFFER", "欄位與快照不同 —— 未能逐格比對"),
                       ("SHAPE-DIFFER", "列數與快照不同 —— 未能逐格比對"),
                       ("READ-ERROR", "無法讀取"),
                       ("NOT-REGENERATED", "不是最近一次執行產生的 —— 請以 run_all.py --clean 重跑"),
                       ("TEXT-DIFFER", "文字欄(標籤/排序)不同"),
                       ("FAIL", "數值超出容差或 NaN 位置不一致")]:
        k = int(counts.get(label, 0))
        if k:
            print(f"★ {k} 個檔 {label}:{why}")
    if n_warn:
        print(f"△ {n_warn} 個檔之 XGBoost/LSTM 相依結果與快照不同(跨平台模式列為警告,不計入失敗;"
              "跨作業系統之可重現性只及於確定性結果)。")
    for m in hard_msgs:
        print(f"★ {m}")

    failed = bool(n_bad or bad_expect or hard_msgs)
    prefix = "【形式檢查,非可重現性證據】" if (stale and a.allow_stale) else ""
    if failed:
        print(f"\n結論:{prefix}未通過。")
        sys.exit(1)
    if a.cross_platform:
        print(f"\n結論:{prefix}通過(跨平台模式)——逐格比對 {n_cmp} 個檔,確定性內容全數在容差內;"
              f"XGBoost/LSTM 相依之差異 {n_warn} 個檔僅列警告。")
    else:
        print(f"\n結論:{prefix}通過——逐格比對 {n_cmp} 個檔(快照清單 {len(ref_list)} 個),"
              "全部分級皆在容差內,勝過基準之病患數與快照相同。")
    sys.exit(0)


if __name__ == "__main__":
    main()
