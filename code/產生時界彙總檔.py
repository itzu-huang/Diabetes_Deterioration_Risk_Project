# -*- coding: utf-8 -*-
"""
產生時界彙總檔.py —— 由三個時界之逐筆結果彙總為 lstm_pooled_by_horizon.csv
================================================================================
【本檔之來歷 — 請務必讀完再使用】
  cgm_output/lstm_pooled_by_horizon.csv 彙總跨病患合併訓練 LSTM 於 PH=15/30/60 分之表現;
  報告第伍章表 5-4-2、表 5-4-3 與相關正文之 30/60 分鐘數字(例如 12.59/22.69 mg/dL、
  R² 0.927/0.762、PH=30 分改善中位數 23.6%、PH=60 分逐紀錄平均 RMSE 22.02)取自本檔;
  make_figures_v2.py 之 slide_lstm_horizons.png 亦由本檔繪製(該圖未嵌入任何交付物)。
  原交付之 `_修正腳本/` 目錄中並未包含產生該檔的任何腳本。亦即:該 CSV 存在、被交付物
  引用,卻沒有任何程式可以重新產生它 —— 這是重現鏈上的斷點。
  〔稽核修正 C-07〕原寫本檔為「報告表 4-3-2、簡報第 15 張與海報『多時界比較』之唯一數值
    來源」不成立:報告無表 4-3-2,簡報第 15 張為二維風險分層,海報無「多時界比較」。原另寫
    「`_修正腳本/README` 宣告 52 支,實交付 32 支,整個 `02_數值統一/` 層 9 支全缺」,於封包內
    查無出處(B 之全部文字檔與巢狀 zip 均無「52 支」之原始宣告;B 之 _腳本說明.txt 寫 29 支、
    資料夾實有 32 支 .py),已刪除該歸因。

  本檔由稽核方以「逆向比對」補寫:讀入三個時界之逐筆輸出,嘗試各種彙總口徑,
  取能完全重現原 CSV 者。驗證結果:本檔產生之 7 欄 × 3 列共 21 格,與原交付
  之 lstm_pooled_by_horizon.csv 逐格相同(見下方 --verify)。
  〔稽核修正 CODE-21〕現行輸出為 12 欄 × 3 列:[修正 35] 另新增 n_test_windows、pooled_rmse、
    r2、baseline_pooled_rmse、baseline_r2 五欄,為重建版新算,不在上述「21 格逐格相同」之
    驗證範圍內(表 5-4-3 之 12.59/22.69/0.927/0.762 即出自新增欄)。

  ★ 但「能重現」不等於「即為原作法」。彙總口徑(平均/中位數、round(x, 3)、
    beat_pct 以 skill_pct > 0 認定)是由結果反推的,原作者是否確實如此撰寫
    無從查證。使用本檔時應在方法章節註明此檔為重建版本。
  〔稽核修正 A-19〕為避免報告自 3 位小數再進位(雙重進位),現改為全精度輸出(不再 round);
    上述「round(x, 3)」僅為逆向重建原檔時之口徑。LSTM 已重新訓練,原交付之 CSV 不再適用。

【輸入】
  cgm_output/lstm_pooled_vs_baseline.csv   PH=15 分(next-step,cgm_lstm_markov.py 產生)
  cgm_output/lstm_pooled_PH30m.csv         PH=30 分(pooled_ph.py 產生;舊檔名 _pooled_ph30m.csv 亦可讀)
  cgm_output/lstm_pooled_PH60m.csv         PH=60 分(pooled_ph.py 產生;舊檔名 _pooled_ph60m.csv 亦可讀)
  〔稽核修正 CODE-21〕原列舊檔名 _pooled_ph30m/60m.csv;[修正 25a] 後已改名。

【輸出】
  cgm_output/lstm_pooled_by_horizon.csv

【用法】
  python 產生時界彙總檔.py             # 產生彙總檔
  python 產生時界彙總檔.py --verify    # 產生後與既有檔逐格比對(不覆寫)
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
import os
import sys

import numpy as np
import pandas as pd

OUTDIR = "cgm_output"
DEST = f"{OUTDIR}/lstm_pooled_by_horizon.csv"

# (PH 分鐘, 來源檔) —— PH=15 之來源另有 n_points / n_train_windows 兩欄,
# 彙總時不使用,故三個時界可共用同一段程式。
def _pick(*cands):
    """回傳第一個存在的檔;都不存在則回傳第一個(供錯誤訊息顯示)。"""
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


# [修正 25a] pooled_ph.py 之輸出檔名已統一為 lstm_pooled_PH{N}m.csv
# (與變更說明_v2.md 及 reference/ 一致);舊檔名 _pooled_ph{N}m.csv 仍可讀。
SOURCES = [
    (15, f"{OUTDIR}/lstm_pooled_vs_baseline.csv"),
    (30, _pick(f"{OUTDIR}/lstm_pooled_PH30m.csv", f"{OUTDIR}/_pooled_ph30m.csv")),
    (60, _pick(f"{OUTDIR}/lstm_pooled_PH60m.csv", f"{OUTDIR}/_pooled_ph60m.csv")),
]

ND = None   # 〔稽核修正 A-19〕原為 3(與原交付檔一致之 round(x, 3));None = 全精度,不捨入


def _rd(x, nd=None):
    """〔稽核修正 A-19〕ND 為 None 時回傳全精度浮點數;否則 round 至 nd 位(nd 未給則用 ND)。"""
    n = ND if nd is None else nd
    return float(x) if ND is None else round(float(x), n)


# ---------------------------------------------------------------------------
# [修正 35] 補「與文獻同口徑」之彙總指標。
#   Alredaini 等(2026, Sci Rep 16:2421)表 9 報 MAE/MAPE/R²/RMSE,其 RMSE 係
#   在測試集上一次算出(全視窗彙總);本檔原有之 lstm_rmse_mean 則是「逐紀錄先算
#   RMSE 再取算術平均」,兩者是不同的估計量,不宜直接並列。
#   本段另外算出「全測試視窗彙總」之 RMSE 與 R²,使報告表 5-4-3 能同口徑對照。
#
#   關鍵:不重跑深度學習。逐紀錄之 RMSE 已在來源檔中,而每筆紀錄之測試視窗數與
#   其對應之實際值完全由資料與切分規則決定(pooled_resumable.py / pooled_ph.py
#   之 n_tr=int(len(g)*0.8)、look_back=12、make_windows 取第 STEP 步),故可精確
#   重建:
#       n_te   = len(g) - n_tr - STEP + 1
#       實際值 = g[n_tr+STEP-1 : n_tr+STEP-1+n_te]
#       SSE    = Σ n_te_i × rmse_i²
#       pooled_rmse = sqrt(SSE / Σ n_te_i)
#       R²     = 1 - SSE / Σ(y - ȳ)²      (ȳ 為全體測試實際值之平均)
#   重跑 LSTM 會使所有深度學習數值改變(見報告研究限制第十一項),故此處刻意
#   以「既有預測誤差 + 可重建之樣本結構」計算,不觸發任何再訓練。
CGM_DIR = "Shanghai_T2DM"
LOOK_BACK = 12


def _load_cgm(fp):
    """與 cgm_lstm_markov.load_one 相同之讀法(此處不 import 該檔以免載入
    TensorFlow;欄位判定邏輯逐字相同)。"""
    df = pd.read_excel(fp)
    date_col = [c for c in df.columns if str(c).strip().lower().startswith("date")][0]
    cgm_col = [c for c in df.columns if "cgm" in str(c).lower()][0]
    t = df[[date_col, cgm_col]].copy()
    t.columns = ["ts", "cgm"]
    t["ts"] = pd.to_datetime(t["ts"], errors="coerce")
    t["cgm"] = pd.to_numeric(t["cgm"], errors="coerce")
    return t.dropna().sort_values("ts").reset_index(drop=True)


_SERIES_CACHE = {}


def _series(rid):
    if rid in _SERIES_CACHE:
        return _SERIES_CACHE[rid]
    import glob as _g
    hit = [f for f in (_g.glob(os.path.join(CGM_DIR, rid + ".xlsx"))
                       + _g.glob(os.path.join(CGM_DIR, rid + ".xls")))]
    if not hit:
        return None
    v = _load_cgm(hit[0])["cgm"].values.astype("float32")
    _SERIES_CACHE[rid] = v
    return v


def pooled_metrics(ph, D):
    """回傳 (n_test_windows, pooled_rmse, r2, baseline_pooled_rmse, baseline_r2)。
    任一紀錄之 CGM 檔缺失即回傳 None(表示無法計算,不以近似值充數)。"""
    step = ph // 15
    n_tot = 0
    sse_l = sse_b = 0.0
    ys = []
    for _, row in D.iterrows():
        g = _series(str(row["record"]))
        if g is None:
            return None
        n_tr = int(len(g) * 0.8)
        n_te = len(g) - n_tr - step + 1
        if n_te <= 0:
            return None
        y = g[n_tr + step - 1: n_tr + step - 1 + n_te].astype("float64")
        if len(y) != n_te:
            return None
        n_tot += n_te
        sse_l += n_te * float(row["lstm_rmse"]) ** 2
        sse_b += n_te * float(row["baseline_rmse"]) ** 2
        ys.append(y)
    yy = np.concatenate(ys)
    sst = float(np.sum((yy - yy.mean()) ** 2))
    return (n_tot,
            float(np.sqrt(sse_l / n_tot)), 1.0 - sse_l / sst,
            float(np.sqrt(sse_b / n_tot)), 1.0 - sse_b / sst)

# ---------------------------------------------------------------------------


def summarize(ph, path):
    """單一時界之彙總。

    口徑(由逆向比對確定,見檔頭說明):
      baseline_rmse_mean  持續性基準 RMSE 之「算術平均」
      lstm_rmse_mean      LSTM RMSE 之「算術平均」
      lstm_rmse_median    LSTM RMSE 之「中位數」
      lstm_mae_mean       LSTM MAE 之「算術平均」
      skill_median        技巧分數 skill_pct 之「中位數」(非平均)
      beat_pct            skill_pct > 0 之筆數占比 × 100(嚴格大於,不含等於 0)
    原全部以 round(x, 3) 輸出;〔稽核修正 A-19〕現為全精度(見 ND)。
    """
    D = pd.read_csv(path, float_precision="round_trip")
    need = ["baseline_rmse", "lstm_rmse", "lstm_mae", "skill_pct"]
    miss = [c for c in need if c not in D.columns]
    if miss:
        sys.exit(f"[錯誤] {path} 缺少欄位:{miss}")
    out = dict(
        PH_min=ph,
        baseline_rmse_mean=_rd(D.baseline_rmse.mean()),
        lstm_rmse_mean=_rd(D.lstm_rmse.mean()),
        lstm_rmse_median=_rd(D.lstm_rmse.median()),
        lstm_mae_mean=_rd(D.lstm_mae.mean()),
        skill_median=_rd(D.skill_pct.median()),
        beat_pct=_rd((D.skill_pct > 0).mean() * 100),
    )
    pm = pooled_metrics(ph, D)
    if pm is None:
        print(f"  [警告] PH={ph} 無法重建測試視窗結構(CGM 檔缺失),"
              f"同口徑欄位留空。")
        out.update(n_test_windows=-1, pooled_rmse=np.nan, r2=np.nan,
                   baseline_pooled_rmse=np.nan, baseline_r2=np.nan)
    else:
        n_tot, prm, r2, bprm, br2 = pm
        out.update(n_test_windows=int(n_tot),
                   pooled_rmse=_rd(prm), r2=_rd(r2, 4),
                   baseline_pooled_rmse=_rd(bprm),
                   baseline_r2=_rd(br2, 4))
    return out, len(D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="與既有 lstm_pooled_by_horizon.csv 逐格比對,不覆寫")
    a = ap.parse_args()

    rows, ns = [], []
    for ph, p in SOURCES:
        if not os.path.exists(p):
            sys.exit(f"[錯誤] 找不到來源檔 {p}。\n"
                     f"       PH=15 由 cgm_lstm_markov.py 產生;"
                     f"PH=30/60 由 pooled_ph.py 產生(需先執行)。")
        r, n = summarize(ph, p)
        rows.append(r)
        ns.append(n)

    R = pd.DataFrame(rows)
    if len(set(ns)) != 1:
        print(f"[警告] 三個時界之筆數不一致:{dict(zip([s[0] for s in SOURCES], ns))}"
              f" —— 時界間之比較將不是同一組病患,請查明。")
    print(f"三個時界各 {ns[0]} 筆(同一組紀錄)")
    print(R.to_string(index=False))

    if a.verify:
        if not os.path.exists(DEST):
            sys.exit(f"[錯誤] 找不到 {DEST},無從比對。")
        G = pd.read_csv(DEST, float_precision="round_trip")
        if list(G.columns) != list(R.columns):
            sys.exit(f"[★] 欄位不同:\n  既有 {list(G.columns)}\n  本檔 {list(R.columns)}")
        if G.shape != R.shape:   # 〔稽核修正 CODE-26〕列數不同亦須判為不符
            sys.exit(f"[★] 列數不同:既有 {G.shape[0]} 列、本檔 {R.shape[0]} 列")
        bad = []
        for c in R.columns:
            a, b = R[c].to_numpy(float), G[c].to_numpy(float)
            # 〔稽核修正 CODE-26〕原以 d.max() > 0 判定,任一側為 NaN 時 d.max() 為 NaN、比較恆為
            #   False,NaN 不一致會被當成「相同」。改為另行比對 NaN 位置。
            nan_bad = np.isnan(a) != np.isnan(b)
            d = np.abs(a - b)
            dmax = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
            if nan_bad.any() or dmax > 0:
                bad.append((c, float("nan") if nan_bad.any() else dmax))
        if bad:
            print("\n★ 以下欄位不符:")
            for c, m in bad:
                print(f"   {c}:最大差 {m:.6g}")
            sys.exit(1)
        print(f"\n✓ 與既有 {DEST} 之 {R.shape[0]}×{R.shape[1]} = {R.size} 格全數相同(差 0)。")
        return

    R.to_csv(DEST, index=False, encoding="utf-8-sig")
    print(f"\n已輸出 → {DEST}")


if __name__ == "__main__":
    main()
