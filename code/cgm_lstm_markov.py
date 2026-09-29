# -*- coding: utf-8 -*-
"""
CGM 時間序列分析:血糖波動圖 + 馬可夫轉移矩陣 + LSTM 血糖預測
============================================================
輸入:Shanghai_T2DM/ 資料夾內 109 個 CGM 檔(每檔一位病患某次監測,每 15 分鐘一筆)。
本檔產出三塊,對應你們報告「LSTM-Markov 動態模型」的概念:

  1) CGM 指標(每筆監測):平均血糖、CV、GMI、TIR / TBR / TAR
     → 存成 cgm_metrics.csv,可用 Patient Number 併回 Shanghai_T2DM_Summary,
       當作額外特徵或分層依據(這是把 CGM 接進主模型的橋樑)。
  2) 血糖狀態馬可夫轉移矩陣(3 態與 5 態)+ 熱圖 + 長期穩態分布。
  3) LSTM 逐步血糖預測(用過去 N 筆預測下一筆)+ 實際 vs 預測「血糖波動圖」。

環境:Colab / Jupyter。需要 pandas, numpy, matplotlib, scikit-learn, tensorflow。
讀 .xls 需要 xlrd:  pip install xlrd tensorflow
(若你已先用 LibreOffice 把 .xls 轉成 .xlsx,就不需要 xlrd。)

固定 random_state=42;輸出寫入 ./cgm_output/。
〔稽核修正 B-07〕亂數種子之實際作法:原程式只設 np.random.seed 與 tf.random.set_seed,
  但 Keras 3 之權重初始化種子取自 Python 內建 random(本專案從未設定),故每次執行之
  初始權重都不同,「固定種子」並未生效。現改為每個 LSTM 函式開始時呼叫
  random.seed(SEED) + keras.utils.set_random_seed(SEED)(同時設定 Python、NumPy、TF 與
  Keras 全域種子),並開啟 tf.config.experimental.enable_op_determinism()。
  實測(同一台機器、兩個獨立行程):初始權重、訓練後權重、預測值與輸出 CSV 逐位元相同。
  ※ 不同作業系統/CPU 之間仍可能不同(浮點運算路徑不同),故報告之 LSTM 數字應以
    學生本機(Windows)那一次完整執行為準。
〔稽核修正 C-02〕圖 4-7-5(lstm_forecast_2000.png)改由 lstm_forecast_all() 中該筆紀錄
  之同一個模型繪製,圖上 RMSE 即等於 lstm_vs_baseline_all.csv 之值;main() 不再另外
  單獨訓練一個 epochs=40 之模型。
〔稽核修正 C-03〕圖 4-7-1(cgm_metrics_dist.png)改為六格(平均血糖、CV、GMI、TIR、
  TBR、TAR),與表 4-7-1 及正文「六項」一致。
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
import glob
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

SEED = 42
np.random.seed(SEED)
# 〔稽核修正 B-07〕Python 內建 random 亦須設定:Keras 3 之權重初始化種子取自它。
import random
random.seed(SEED)

CGM_DIR = "Shanghai_T2DM"          # 放 109 個 CGM 檔的資料夾(.xls 或 .xlsx 皆可)
OUTDIR = "cgm_output"
os.makedirs(OUTDIR, exist_ok=True)

# 〔稽核修正 B-16〕本程式之 LSTM 段落所產生之檔案。沒有 tensorflow 又未設 SKIP_LSTM 時,
#   main() 會刪除這些舊檔並以非 0 結束,不讓上一次執行留下的 LSTM 結果冒充本次結果。
LSTM_OUTPUTS = ["lstm_vs_baseline_all.csv", "lstm_vs_baseline_all.png",
                "lstm_pooled_vs_baseline.csv", "lstm_pooled_vs_perpatient.png",
                "lstm_forecast_2000.png"]


class TensorFlowMissing(RuntimeError):
    """〔稽核修正 B-16〕只有 import tensorflow/keras 發生 ImportError(未安裝,或 Windows 上
    「DLL load failed」這類載入失敗)時才會拋出此例外,訊息內附原始錯誤;其他種類之例外
    (例如設定種子或建模時之錯誤)一律原樣拋出,不再被當成「沒有 tensorflow」而吞掉。"""


def _keras_setup(seed=None):
    """〔稽核修正 B-07〕〔稽核修正 B-16〕載入 tensorflow/keras 並固定全部亂數來源。
    每個 LSTM 函式開始時各呼叫一次,使該函式之結果只取決於(資料、設定、SEED),
    與之前是否跑過其他函式無關。seed=None 時用 SEED;pooled_ph.py 於每折開始時
    以 SEED+折號 呼叫,使「中斷後續跑」與「一次跑完」之結果相同。

    - random.seed + keras.utils.set_random_seed:後者同時設定 Python random、NumPy、
      TF 全域種子並重設 Keras 之全域 SeedGenerator(權重初始化、Dropout、洗牌)。
    - tf.config.experimental.enable_op_determinism():要求 TF 只用確定性之運算實作;
      若某運算沒有確定性實作會直接報錯,而不是默默產生不可重現之結果。
      實測(Linux、CPU):開啟與否結果逐位元相同、速度差異在量測誤差內
      (見 _測試紀錄/CODE/B-07_種子/)。
    """
    try:
        import tensorflow as tf
        import keras
    except ImportError as e:   # 只有「沒安裝」才當成沒有 tensorflow;其他例外照常拋出
        raise TensorFlowMissing(
            "找不到 tensorflow/keras(ImportError: %s)。LSTM 段落需要 tensorflow;"
            "請執行 pip install -r requirements.txt,或以 run_all.py --skip-lstm "
            "(設定 SKIP_LSTM=1)明確略過。" % e) from e
    s = SEED if seed is None else int(seed)
    random.seed(s)
    keras.utils.set_random_seed(s)
    tf.config.experimental.enable_op_determinism()
    return tf, keras


def build_lstm_model(look_back=12, n_out=1, units=64):
    """〔稽核修正 B-12〕全專案 LSTM 之唯一架構定義(逐人版、合併版、pooled_ph.py、
    make_diagnostics.py 之參數量皆由此而來):LSTM(units) → Dropout(0.2) → Dense(n_out)。
    units=64、look_back=12、n_out=1 時參數量為 16,961(LSTM 16,896 + Dense 65)。
    呼叫前須先 _keras_setup()(或自行 import tensorflow)。"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    return Sequential([
        LSTM(units, input_shape=(look_back, 1)),
        Dropout(0.2),
        Dense(n_out),
    ])


# ------------------------------------------------------------------ #
# 1. CGM 讀取(自動處理 .xls 與 .xlsx)
# ------------------------------------------------------------------ #
def load_one(fp):
    """讀單一 CGM 檔,回傳 [ts, cgm] 已排序、去空值的 DataFrame。"""
    df = pd.read_excel(fp)   # pandas 會自動選 engine;.xls 需已安裝 xlrd
    date_col = [c for c in df.columns if str(c).strip().lower().startswith("date")][0]
    cgm_col = [c for c in df.columns if "cgm" in str(c).lower()][0]
    s = df[[date_col, cgm_col]].copy()
    s.columns = ["ts", "cgm"]
    s["ts"] = pd.to_datetime(s["ts"], errors="coerce")
    s["cgm"] = pd.to_numeric(s["cgm"], errors="coerce")
    return s.dropna().sort_values("ts").reset_index(drop=True)


def list_cgm_files():
    files = sorted(glob.glob(os.path.join(CGM_DIR, "*.xlsx")) +
                   glob.glob(os.path.join(CGM_DIR, "*.xls")))
    if not files:
        raise FileNotFoundError(f"在 {CGM_DIR}/ 找不到 CGM 檔。請確認資料夾路徑。")
    return files


def patient_id_of(fp):
    return os.path.basename(fp).split("_")[0]


# ------------------------------------------------------------------ #
# 2. CGM 指標(TIR / TBR / TAR / GMI / CV)
#    國際共識目標範圍 70–180 mg/dL
# ------------------------------------------------------------------ #
def cgm_metrics(g):
    g = np.asarray(g, float)
    return dict(
        n=len(g),
        mean=g.mean(),
        SD=g.std(),
        CV=g.std() / g.mean() * 100,
        GMI=3.31 + 0.02392 * g.mean(),         # GMI(%) ← 平均血糖(mg/dL)
        TIR=np.mean((g >= 70) & (g <= 180)) * 100,   # Time In Range
        TBR=np.mean(g < 70) * 100,                   # Time Below Range
        TAR=np.mean(g > 180) * 100,                  # Time Above Range
    )


def build_metrics_table():
    rows = []
    for fp in list_cgm_files():
        s = load_one(fp)
        if len(s) < 10:
            continue
        m = cgm_metrics(s["cgm"].values)
        m["Patient_Number"] = os.path.splitext(os.path.basename(fp))[0]  # 例 2000_0_20201230
        m["patient"] = patient_id_of(fp)
        rows.append(m)
    met = pd.DataFrame(rows)
    met.to_csv(f"{OUTDIR}/cgm_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"[metrics] {len(met)} 筆監測 → cgm_output/cgm_metrics.csv")
    print(met[["mean", "CV", "GMI", "TIR", "TBR", "TAR"]].describe().round(1).to_string())
    return met


# ------------------------------------------------------------------ #
# 3. 血糖狀態馬可夫轉移矩陣
# ------------------------------------------------------------------ #
STATE_DEFS = {
    3: (["Low(<70)", "InRange(70-180)", "High(>180)"],
        lambda v: 0 if v < 70 else (1 if v <= 180 else 2)),
    5: (["VLow(<54)", "Low(54-70)", "InRange(70-180)", "High(180-250)", "VHigh(>250)"],
        lambda v: 0 if v < 54 else (1 if v < 70 else (2 if v <= 180 else (3 if v <= 250 else 4)))),
}


def markov_matrix(n_states=3, max_gap_min=30):
    """
    以「相鄰兩筆 CGM」的狀態轉移建立母體轉移矩陣。
    max_gap_min:若兩筆間隔超過此分鐘數(資料中斷)則不計該轉移,避免跨斷點的假轉移。
    """
    labels, f = STATE_DEFS[n_states]
    T = np.zeros((n_states, n_states))
    for fp in list_cgm_files():
        s = load_one(fp)
        if len(s) < 10:
            continue
        st = s["cgm"].map(f).values
        gap = s["ts"].diff().dt.total_seconds().div(60).values  # 相鄰間隔(分)
        for i in range(1, len(st)):
            if np.isfinite(gap[i]) and gap[i] <= max_gap_min:
                T[st[i - 1], st[i]] += 1
    P = np.divide(T, T.sum(axis=1, keepdims=True),
                  out=np.zeros_like(T), where=T.sum(axis=1, keepdims=True) > 0)

    # 長期穩態分布(轉移矩陣的左特徵向量,特徵值=1)
    vals, vecs = np.linalg.eig(P.T)
    stat = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    stat = stat / stat.sum()

    # 熱圖
    fig, ax = plt.subplots(figsize=(1.6 * n_states + 1, 1.4 * n_states + 1))
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_states)); ax.set_yticks(range(n_states))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("To state"); ax.set_ylabel("From state")
    ax.set_title(f"Glucose-state transition matrix ({n_states}-state)")
    for i in range(n_states):
        for j in range(n_states):
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center",
                    color="white" if P[i, j] > 0.5 else "black", fontsize=9)
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/markov_{n_states}state.png", dpi=150); _show()

    # [修正 19] 原輸出第一欄(狀態名)無標頭,讀檔者無從判斷列是「自」或「至」。
    #   轉移矩陣 P[i][j] = P(下一時點為 j | 現在為 i),故列為「自」、欄為「至」。
    pd.DataFrame(P, index=labels, columns=labels).to_csv(
        f"{OUTDIR}/markov_{n_states}state.csv",
        index_label="from_state（列=自；欄=至）", encoding="utf-8-sig")
    print(f"\n[markov] {n_states}-state 轉移矩陣 → cgm_output/markov_{n_states}state.png / .csv")
    print("  states:", labels)
    print("  P=\n", np.round(P, 3))
    print("  長期穩態分布(≈各狀態時間占比):", dict(zip(labels, np.round(stat, 3))))
    return P, labels, stat


def markov_convergence(P, labels, start_state="InRange", steps=None, tag="",
                       step_min=15):
    """
    輸出馬可夫鏈之收斂診斷:穩態分布、第二大特徵值、半衰期,以及 P^k 之逐步分布。

    為何需要:短期風險軸以 P^k 推算「未來 k 步之出範圍機率」。
    但轉移矩陣之 k 次方會隨 k 增大收斂至穩態,收斂後該列與「起點為何狀態」已無關,
    其數值必然等同該段監測之狀態時間占比(即 TIR/TBR/TAR)。
    因此長時界之推算不具前瞻意義,本表即為此性質之直接證據,
    亦為短期風險軸選擇時界之量化依據。

    參數
      P            : 轉移矩陣(markov_matrix 之第一個回傳值)
      labels       : 狀態名稱清單
      start_state  : 起點狀態(預設目標範圍內)
      steps        : 欲檢視之時步;預設 15 分鐘至 48 小時
      tag          : 輸出檔名後綴(例如 "_3state")
      step_min     : 每一時步之分鐘數(CGM 取樣間隔,本資料為 15 分鐘)

    輸出
      cgm_output/markov_convergence{tag}.csv  逐步分布與距穩態之差距
      cgm_output/markov_stationary{tag}.csv   穩態分布、|lambda2| 與半衰期
    """
    n = P.shape[0]
    if steps is None:
        steps = [1, 2, 4, 8, 16, 24, 32, 48, 96, 192]
    si = next((i for i, l in enumerate(labels)
               if str(l).lower().startswith(start_state.lower())), n // 2)

    # 穩態分布:轉移矩陣之左特徵向量(特徵值 = 1)
    vals, vecs = np.linalg.eig(P.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    pi = pi / pi.sum()
    # 收斂速度:第二大特徵值之模與其對應半衰期
    ev = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    lam2 = float(ev[1]) if n > 1 else 0.0
    half_steps = np.log(0.5) / np.log(lam2) if 0 < lam2 < 1 else float("nan")

    # 〔稽核修正 A-19〕原寫檔前先 round(百分比 2 位、lambda2 4 位、半衰期 2 位),
    #   報告再從 CSV 進位一次即成「雙重進位」:|λ2| 全精度 0.887474 → CSV 0.8875 → 報告 0.888
    #   (正確應為 0.887)。現改為全精度寫出(數值本身不變,只是不再先捨入),由報告端
    #   依需要之位數一次進位。hours 為 k×15/60,本身即為精確值,維持原寫法。
    rows = []
    for k in steps:
        v = np.linalg.matrix_power(P, k)[si]
        rows.append({"k_steps": k, "hours": round(k * step_min / 60, 2),
                     **{str(l): float(x) * 100 for l, x in zip(labels, v)},
                     "max_dev_from_stationary_pp": float(np.abs(v - pi).max()) * 100})
    rows.append({"k_steps": "stationary", "hours": "inf",
                 **{str(l): float(x) * 100 for l, x in zip(labels, pi)},
                 "max_dev_from_stationary_pp": 0.0})
    C = pd.DataFrame(rows)
    C.to_csv(f"{OUTDIR}/markov_convergence{tag}.csv", index=False, encoding="utf-8-sig")

    S = pd.DataFrame([{**{str(l): float(x) * 100 for l, x in zip(labels, pi)},
                       "lambda2": lam2,
                       "half_life_steps": float(half_steps),
                       "half_life_hours": float(half_steps) * step_min / 60,
                       "start_state": labels[si]}])
    S.to_csv(f"{OUTDIR}/markov_stationary{tag}.csv", index=False, encoding="utf-8-sig")

    print(f"\n[markov] 收斂診斷{tag}(自 {labels[si]} 出發)"
          f" → markov_convergence{tag}.csv / markov_stationary{tag}.csv")
    print("  穩態分布:", dict(zip(labels, np.round(pi * 100, 2))))
    print(f"  |lambda2| = {lam2:.4f} → 半衰期 {half_steps:.1f} 步"
          f" = {half_steps * step_min / 60:.1f} 小時")
    print(C.to_string(index=False))
    print("  ※ 當『與穩態差距』趨近 0,該時界之推算已等同狀態時間占比,不再帶有起點資訊。")
    return C, S


# ------------------------------------------------------------------ #
# 4. 血糖波動圖(單一病患,含目標範圍帶)
# ------------------------------------------------------------------ #
def plot_fluctuation(patient_prefix=None, n_examples=3):
    """畫個別病患血糖波動圖。n_examples=None 表示畫全部 109 位(檔案會很多)。"""
    files = list_cgm_files()
    if patient_prefix:
        files = [f for f in files if os.path.basename(f).startswith(str(patient_prefix))]
    if n_examples is not None:
        files = files[:n_examples]
    for fp in files:
        s = load_one(fp)
        name = os.path.splitext(os.path.basename(fp))[0]
        plt.figure(figsize=(11, 3.2))
        plt.axhspan(70, 180, color="#86efac", alpha=0.35, label="Target 70–180")
        plt.axhline(70, color="#ca8a04", lw=0.8, ls="--")
        plt.axhline(180, color="#ca8a04", lw=0.8, ls="--")
        plt.plot(s["ts"], s["cgm"], color="#1d4ed8", lw=0.9)
        m = cgm_metrics(s["cgm"].values)
        plt.ylabel("CGM (mg/dL)")
        plt.title(f"CGM fluctuation — {name}  (TIR={m['TIR']:.0f}%, GMI={m['GMI']:.1f}, CV={m['CV']:.0f}%)")
        plt.legend(loc="upper right", fontsize=8); plt.tight_layout()
        plt.savefig(f"{OUTDIR}/fluctuation_{name}.png", dpi=150); _show()
    print(f"\n[fluctuation] 已輸出 {len(files)} 張血糖波動圖 → cgm_output/fluctuation_*.png")


# ------------------------------------------------------------------ #
# 4b. 合併視圖:AGP(日內百分位)+ 示範網格 + 指標分布
# ------------------------------------------------------------------ #
def plot_agp():
    """把全部監測依『一天中的時刻』聚合,畫中位數 + IQR + 5–95% 帶(標準 AGP 圖)。"""
    rows = []
    for fp in list_cgm_files():
        s = load_one(fp)
        if len(s) < 10:
            continue
        tmin = (s["ts"].dt.hour * 60 + s["ts"].dt.minute) // 15 * 15
        rows.append(pd.DataFrame({"tmin": tmin.values, "cgm": s["cgm"].values}))
    A = pd.concat(rows, ignore_index=True)
    grid = np.arange(0, 1440, 15)
    q = A.groupby("tmin")["cgm"].quantile([.05, .25, .5, .75, .95]).unstack().reindex(grid)
    x = grid / 60.0
    plt.figure(figsize=(11, 4))
    plt.axhspan(70, 180, color="#86efac", alpha=0.30, label="Target 70–180")
    plt.fill_between(x, q[.05], q[.95], color="#93c5fd", alpha=0.35, label="5–95%")
    plt.fill_between(x, q[.25], q[.75], color="#3b82f6", alpha=0.45, label="25–75% (IQR)")
    plt.plot(x, q[.5], color="#1e3a8a", lw=2, label="Median")
    plt.axhline(70, color="#ca8a04", lw=0.8, ls="--"); plt.axhline(180, color="#ca8a04", lw=0.8, ls="--")
    plt.xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
    plt.xlim(0, 24); plt.xlabel("Time of day"); plt.ylabel("CGM (mg/dL)")
    plt.title("Ambulatory Glucose Profile (AGP) — all recordings pooled")
    plt.legend(loc="upper right", fontsize=8, ncol=2); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/AGP_all.png", dpi=150); _show()
    print(f"[AGP] cgm_output/AGP_all.png")


def plot_fluctuation_grid(n=12):
    files = list_cgm_files()[:n]
    ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.5, nrow * 2.3), sharey=True)
    for ax, fp in zip(axes.ravel(), files):
        s = load_one(fp); name = os.path.basename(fp).split("_")[0]
        ax.axhspan(70, 180, color="#86efac", alpha=0.25)
        ax.plot(s["ts"], s["cgm"], color="#1d4ed8", lw=0.6)
        tir = np.mean((s["cgm"] >= 70) & (s["cgm"] <= 180)) * 100
        ax.set_title(f"{name}  TIR={tir:.0f}%", fontsize=9)
        ax.tick_params(labelbottom=False, labelsize=7)
    for ax in axes.ravel()[len(files):]:
        ax.axis("off")
    fig.suptitle(f"CGM fluctuation — {n} example patients", y=1.01)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/fluctuation_grid{n}.png", dpi=150, bbox_inches="tight"); _show()
    print(f"[grid] cgm_output/fluctuation_grid{n}.png")


def plot_metrics_distribution(met):
    # 〔稽核修正 C-03〕原圖只有 TIR、GMI、CV 三格,報告正文(¶608)卻稱「六項血糖控制指標之
    #   分布」並描述圖中不存在之 TBR 分布。改為與表 4-7-1 相同之六項、同一順序(2 列 × 3 格),
    #   並加畫中位數線(正文 ¶605 引用者為中位數)。
    panels = [("mean", "Mean glucose (mg/dL)"), ("CV", "CV (%)"), ("GMI", "GMI (%)"),
              ("TIR", "TIR, 70–180 mg/dL (%)"), ("TBR", "TBR, <70 mg/dL (%)"),
              ("TAR", "TAR, >180 mg/dL (%)")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    for ax, (col, lab) in zip(axes.ravel(), panels):
        ax.hist(met[col], bins=20, color="#0d9488", edgecolor="white")
        ax.axvline(met[col].mean(), color="#dc2626", ls="--", lw=1.2, label=f"mean={met[col].mean():.1f}")
        ax.axvline(met[col].median(), color="#1e293b", ls=":", lw=1.2,
                   label=f"median={met[col].median():.1f}")
        ax.set_title(lab); ax.legend(fontsize=8)
    fig.suptitle(f"Distribution of CGM metrics across recordings (n={len(met)})", y=1.01)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/cgm_metrics_dist.png", dpi=150, bbox_inches="tight"); _show()
    # 〔稽核修正 A-19〕原 .round(2) 後寫檔:例如 CV 中位數全精度 27.448 → CSV 27.45 → 若再依此
    #   進位至 1 位即成 27.5(報告 ¶605 之 27.4 才正確)。改為全精度寫出,由引用端一次進位。
    met[["mean", "CV", "GMI", "TIR", "TBR", "TAR"]].describe().T.to_csv(
        f"{OUTDIR}/cgm_descriptive.csv",
        index_label="metric", encoding="utf-8-sig")   # [修正 19]
    print(f"[dist] cgm_output/cgm_metrics_dist.png / cgm_descriptive.csv")


# ------------------------------------------------------------------ #
# 5. LSTM 逐步血糖預測 + 實際 vs 預測波動圖
# ------------------------------------------------------------------ #
def make_windows(series, look_back, horizon=1):
    # 〔稽核修正 CODE-24〕(僅註明,未改演算法)本函式依「讀值順序」切視窗,不看時間戳;
    #   馬可夫矩陣(markov_matrix、deterioration_risk.transition_matrix)則排除間隔 >30 分鐘
    #   之相鄰配對。兩者對資料斷點之口徑不同。實測 109 檔共 112,178 個相鄰間隔中僅 3 個
    #   >30 分鐘(另 7 個恰 30 分鐘),PH=15 分之測試視窗 22,498 個中 49 個(0.22%)跨越
    #   >15 分鐘之間隔,影響可忽略;若要一致,可改為遇斷點即分段切視窗。
    X, y = [], []
    for i in range(len(series) - look_back - horizon + 1):
        X.append(series[i:i + look_back])
        y.append(series[i + look_back: i + look_back + horizon])
    return np.array(X), np.array(y)


def lstm_forecast(patient_prefix="2000", look_back=12, horizon=1, epochs=40):
    """
    對單一病患做 next-step 血糖預測(look_back=12 ≈ 過去 3 小時 → 預測下一個 15 分鐘)。
    以時間切分:前 80% 訓練、後 20% 測試;標準化只用訓練段統計量(避免洩漏)。
    需要 tensorflow。若要對多位病患,外面包一層迴圈或改成 pooled 訓練即可。

    〔稽核修正 C-02〕main() 已不再呼叫本函式:報告圖 4-7-5 改由 lstm_forecast_all() 中
    同一筆紀錄之同一模型繪製(見該函式 plot_record_prefix)。本函式保留供單獨試驗,
    但輸出檔名改為 lstm_forecast_{病患}_standalone.png,以免覆蓋報告所用之圖。
    """
    _keras_setup()   # 〔稽核修正 B-07／B-16〕固定全部亂數來源;只有 ImportError 才視為缺 tensorflow
    from tensorflow.keras.callbacks import EarlyStopping

    files = [f for f in list_cgm_files() if os.path.basename(f).startswith(str(patient_prefix))]
    if not files:
        # 〔稽核修正 B-16〕原為印一行後 return None(呼叫端無從得知失敗),改為拋出例外。
        raise FileNotFoundError(f"[LSTM] 找不到病患 {patient_prefix} 的 CGM 檔。")
    fp = files[0]
    s = load_one(fp)
    g = s["cgm"].values.astype("float32")
    n_train = int(len(g) * 0.8)

    mu, sd = g[:n_train].mean(), g[:n_train].std()
    gz = (g - mu) / sd
    Xtr, ytr = make_windows(gz[:n_train], look_back, horizon)
    Xte, yte = make_windows(gz[n_train - look_back:], look_back, horizon)  # 保留接續
    Xtr = Xtr.reshape(-1, look_back, 1)
    Xte = Xte.reshape(-1, look_back, 1)

    model = build_lstm_model(look_back, horizon)   # 〔稽核修正 B-12〕與其他 LSTM 共用同一架構定義
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    es = EarlyStopping(patience=8, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_split=0.15, epochs=epochs, batch_size=32,
              callbacks=[es], verbose=0)

    # 測試段預測(取 horizon 的第 1 步),反標準化回 mg/dL
    pred_z = model.predict(Xte, verbose=0)[:, 0]
    pred = pred_z * sd + mu
    true = yte[:, 0] * sd + mu
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))

    # 持續性基準(predict next = current):用每個測試窗的最後一筆(即當前血糖)當預測,
    # 與 LSTM 使用完全相同的測試集,可逐窗公平對照。
    baseline = Xte[:, -1, 0] * sd + mu
    baseline_rmse = float(np.sqrt(np.mean((baseline - true) ** 2)))
    baseline_mae = float(np.mean(np.abs(baseline - true)))
    # 〔稽核修正 CODE-26〕基準 RMSE 為 0 時技巧分數無定義:原本逐人版退回 0.0、合併版退回 NaN,
    #   兩版口徑不一;統一為 NaN(現行資料無基準 RMSE 為 0 之紀錄,不影響任何輸出)。
    skill = (1 - rmse / baseline_rmse) * 100 if baseline_rmse > 0 else np.nan  # RMSE 相對改善(%)

    # 實際 vs 預測 波動圖(測試段)
    ts_test = s["ts"].values[n_train:n_train + len(true)]
    plt.figure(figsize=(11, 3.4))
    plt.axhspan(70, 180, color="#86efac", alpha=0.30)
    plt.plot(ts_test, true, color="#1d4ed8", lw=1.2, label="Actual")
    plt.plot(ts_test, pred, color="#dc2626", lw=1.2, ls="--", label="LSTM predicted")
    plt.plot(ts_test, baseline, color="#6b7280", lw=1.0, ls=":", label="Persistence baseline")
    plt.ylabel("CGM (mg/dL)")
    plt.title(f"LSTM next-step forecast — patient {patient_prefix} (standalone model)  "
              f"(LSTM RMSE={rmse:.1f} vs baseline {baseline_rmse:.1f} mg/dL)")
    plt.legend(loc="upper right", fontsize=8); plt.tight_layout()
    _fn = f"lstm_forecast_{patient_prefix}_standalone.png"   # 〔稽核修正 C-02〕不覆蓋報告圖
    plt.savefig(f"{OUTDIR}/{_fn}", dpi=150); _show()
    print(f"\n[LSTM] 病患 {patient_prefix}:LSTM RMSE={rmse:.1f}/MAE={mae:.1f};"
          f"持續性基準 RMSE={baseline_rmse:.1f}/MAE={baseline_mae:.1f} mg/dL"
          f"(RMSE 相對改善 {skill:.1f}%)→ cgm_output/{_fn}")
    return dict(patient=patient_prefix, rmse=rmse, mae=mae,
                baseline_rmse=baseline_rmse, baseline_mae=baseline_mae, skill_pct=skill)


# ------------------------------------------------------------------ #
# 5b. 全體病患:LSTM vs 持續性基準 比較
# ------------------------------------------------------------------ #
def baseline_all(look_back=12):
    """
    只算「持續性基準」(predict next = current)於全部病患的測試段誤差。
    不需要 tensorflow,且完全確定(無隨機性),可先跑此函式取得基準分布。
    切分方式與 lstm_forecast 完全一致,故可直接與 LSTM 結果對照。
    """
    rows = []
    for fp in list_cgm_files():
        rid = os.path.splitext(os.path.basename(fp))[0]
        g = load_one(fp)["cgm"].values.astype("float32")
        if len(g) < look_back + 20:
            continue
        n_train = int(len(g) * 0.8)
        mu, sd = g[:n_train].mean(), g[:n_train].std()
        gz = (g - mu) / sd
        Xte, yte = make_windows(gz[n_train - look_back:], look_back, 1)
        Xte = Xte.reshape(-1, look_back, 1)
        true = yte[:, 0] * sd + mu
        base = Xte[:, -1, 0] * sd + mu
        rows.append(dict(record=rid, patient=rid.split("_")[0], n_test=len(true),
                         baseline_rmse=float(np.sqrt(np.mean((base - true) ** 2))),
                         baseline_mae=float(np.mean(np.abs(base - true)))))
    B = pd.DataFrame(rows)
    B.to_csv(f"{OUTDIR}/baseline_all.csv", index=False, encoding="utf-8-sig")
    print(f"[baseline] {len(B)} 筆監測 → cgm_output/baseline_all.csv")
    print(f"  持續性基準 RMSE:mean={B.baseline_rmse.mean():.2f}, "
          f"median={B.baseline_rmse.median():.2f}, "
          f"min={B.baseline_rmse.min():.2f}, max={B.baseline_rmse.max():.2f} mg/dL")
    return B


def _plot_record_forecast(rid, prefix, ts_test, true, pred, base, l_rmse, b_rmse):
    """〔稽核修正 C-02〕以 lstm_forecast_all() 中「同一個模型」之測試段預測繪製單筆紀錄之
    實際 vs 預測圖(報告圖 4-7-5、簡報第 14 張所用之 lstm_forecast_{prefix}.png)。
    圖上 RMSE 與 lstm_vs_baseline_all.csv 該列之 lstm_rmse / baseline_rmse 為同一數值。"""
    plt.figure(figsize=(11, 3.4))
    plt.axhspan(70, 180, color="#86efac", alpha=0.30)
    plt.plot(ts_test, true, color="#1d4ed8", lw=1.2, label="Actual")
    plt.plot(ts_test, pred, color="#dc2626", lw=1.2, ls="--", label="LSTM predicted")
    plt.plot(ts_test, base, color="#6b7280", lw=1.0, ls=":", label="Persistence baseline")
    plt.ylabel("CGM (mg/dL)")
    # 標題沿用原圖格式;RMSE 改印 2 位小數,與 lstm_vs_baseline_all.csv 引用位數一致
    plt.title(f"LSTM next-step forecast — patient {prefix}  "
              f"(LSTM RMSE={l_rmse:.2f} vs baseline {b_rmse:.2f} mg/dL)")
    plt.legend(loc="upper right", fontsize=8); plt.tight_layout()
    plt.savefig(f"{OUTDIR}/lstm_forecast_{prefix}.png", dpi=150); _show()
    print(f"[LSTM-all] {rid}:LSTM RMSE={l_rmse:.4f}、基準 RMSE={b_rmse:.4f}"
          f"(與 lstm_vs_baseline_all.csv 同一模型)→ {OUTDIR}/lstm_forecast_{prefix}.png")


def lstm_forecast_all(look_back=12, horizon=1, epochs=30, n_patients=None, verbose=True,
                      plot_record_prefix="2000"):
    """
    對「全部病患」逐一訓練 LSTM 並與持續性基準比較,輸出比較表與圖。
    n_patients:限制數量(例如先設 20 試跑);None = 全部。
    注意:109 位逐一訓練約需數分鐘至數十分鐘,建議先用 n_patients 小量測試。
    plot_record_prefix:〔稽核修正 C-02〕檔名以此開頭之第一筆紀錄(依檔名排序,與原
      lstm_forecast() 之選法相同;"2000" → 2000_0_20201230),以本函式訓練之同一模型輸出
      lstm_forecast_{prefix}.png;None 則不畫。
    """
    _keras_setup()   # 〔稽核修正 B-07／B-16〕固定全部亂數來源;只有 ImportError 才視為缺 tensorflow
    from tensorflow.keras.callbacks import EarlyStopping

    files = list_cgm_files()
    if n_patients:
        files = files[:n_patients]
    if plot_record_prefix is not None and not any(
            os.path.basename(f).startswith(str(plot_record_prefix)) for f in files):
        # 〔稽核修正 B-16〕找不到要畫的紀錄時於訓練前即明確報錯,不讓舊圖冒充本次結果
        raise FileNotFoundError(f"[LSTM-all] 找不到檔名以 {plot_record_prefix} 開頭之紀錄,"
                                f"無法產生 lstm_forecast_{plot_record_prefix}.png")
    rows = []
    plotted = False
    for i, fp in enumerate(files, 1):
        rid = os.path.splitext(os.path.basename(fp))[0]
        s = load_one(fp)
        g = s["cgm"].values.astype("float32")
        if len(g) < look_back + 20:
            continue
        n_train = int(len(g) * 0.8)
        mu, sd = g[:n_train].mean(), g[:n_train].std()
        gz = (g - mu) / sd
        Xtr, ytr = make_windows(gz[:n_train], look_back, horizon)
        Xte, yte = make_windows(gz[n_train - look_back:], look_back, horizon)
        Xtr = Xtr.reshape(-1, look_back, 1); Xte = Xte.reshape(-1, look_back, 1)

        model = build_lstm_model(look_back, horizon)   # 〔稽核修正 B-12〕共用架構定義
        model.compile(optimizer="adam", loss="mse")
        model.fit(Xtr, ytr, validation_split=0.15, epochs=epochs, batch_size=32,
                  callbacks=[EarlyStopping(patience=8, restore_best_weights=True)], verbose=0)

        true = yte[:, 0] * sd + mu
        pred = model.predict(Xte, verbose=0)[:, 0] * sd + mu
        base = Xte[:, -1, 0] * sd + mu
        lr_ = float(np.sqrt(np.mean((pred - true) ** 2)))
        br_ = float(np.sqrt(np.mean((base - true) ** 2)))
        rows.append(dict(record=rid, patient=rid.split("_")[0],
                         lstm_rmse=lr_, lstm_mae=float(np.mean(np.abs(pred - true))),
                         baseline_rmse=br_, baseline_mae=float(np.mean(np.abs(base - true))),
                         # 〔稽核修正 CODE-26〕分母為 0 時與合併版同為 NaN(原為 0.0)
                         skill_pct=(1 - lr_ / br_) * 100 if br_ > 0 else np.nan))
        if (plot_record_prefix is not None and not plotted
                and os.path.basename(fp).startswith(str(plot_record_prefix))):
            ts_test = s["ts"].values[n_train:n_train + len(true)]
            _plot_record_forecast(rid, plot_record_prefix, ts_test, true, pred, base, lr_, br_)
            plotted = True
        if verbose and i % 10 == 0:
            print(f"  ...已完成 {i}/{len(files)}")
    if plot_record_prefix is not None and not plotted:
        # 〔稽核修正 B-16〕該紀錄因資料過短被略過等情形:明確報錯,不留舊圖冒充本次結果
        raise RuntimeError(f"[LSTM-all] 紀錄 {plot_record_prefix}* 未完成訓練,"
                           f"無法產生 lstm_forecast_{plot_record_prefix}.png")

    R = pd.DataFrame(rows)
    R.to_csv(f"{OUTDIR}/lstm_vs_baseline_all.csv", index=False, encoding="utf-8-sig")
    win = (R.lstm_rmse < R.baseline_rmse).mean() * 100

    # 圖 1:散佈(基準 vs LSTM),對角線下方代表 LSTM 較佳
    lim = float(max(R.baseline_rmse.max(), R.lstm_rmse.max())) * 1.05
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(R.baseline_rmse, R.lstm_rmse, s=28, alpha=0.7, color="#dc2626", edgecolor="white")
    ax[0].plot([0, lim], [0, lim], "--", color="grey", lw=1)
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[0].set_xlabel("Persistence baseline RMSE (mg/dL)")
    ax[0].set_ylabel("LSTM RMSE (mg/dL)")
    ax[0].set_title(f"LSTM vs baseline (below line = LSTM better, {win:.0f}%)")
    ax[1].hist(R.skill_pct, bins=20, color="#0d9488", edgecolor="white")
    ax[1].axvline(0, color="grey", ls="--", lw=1)
    ax[1].axvline(R.skill_pct.mean(), color="#dc2626", ls="--", lw=1.2,
                  label=f"mean={R.skill_pct.mean():.1f}%")
    ax[1].set_xlabel("RMSE improvement over baseline (%)"); ax[1].set_ylabel("count")
    ax[1].set_title("Distribution of LSTM skill"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/lstm_vs_baseline_all.png", dpi=150); _show()

    print(f"\n[LSTM-all] {len(R)} 位病患完成 → cgm_output/lstm_vs_baseline_all.csv / .png")
    print(f"  LSTM RMSE     mean={R.lstm_rmse.mean():.2f}  median={R.lstm_rmse.median():.2f}")
    print(f"  基準 RMSE     mean={R.baseline_rmse.mean():.2f}  median={R.baseline_rmse.median():.2f}")
    print(f"  平均相對改善  {R.skill_pct.mean():.1f}%  |  LSTM 勝過基準的比例:{win:.1f}%")
    return R


# ------------------------------------------------------------------ #
# 6. 主程式
# ------------------------------------------------------------------ #
def lstm_forecast_pooled(look_back=12, horizon=1, epochs=30, n_folds=5,
                         batch_size=256, units=64, verbose=True):
    """
    跨病患合併訓練的 LSTM(丙案),並以「病患分組交叉驗證」評估對全新病患的泛化能力。

    與 lstm_forecast_all()(每人各訓練一個模型)的差異:
      - 逐人版:每位病患用自己的前 80% 訓練專屬模型。訓練窗平均僅約 810 個,
               而 LSTM(64) 連同輸出層共 16,961 個參數(〔稽核修正 B-12〕原寫「約 16,900」,
               為手寫近似值;實算見 build_lstm_model().count_params()),序列較短者必然過擬合。
      - 本函式:把全部病患的訓練段合併成一個訓練集(約 88,000 個窗)訓練單一模型,
               參數與樣本比例回到合理範圍。

    驗證設計(關鍵):
      依「病患編號」而非「監測紀錄」分成 n_folds 組(同一病患的多次回診必在同一組)。
      每一折以其餘病患的訓練段配適模型,再預測本折病患的測試段——
      模型完全沒看過這些病患的任何資料,故所得指標為「對全新病患」的泛化表現,
      而非單純的樣本內配適。此為合併訓練必須做的驗證,否則會被質疑模型已見過該病患。

    標準化:各病患仍以「自己訓練段」的平均與標準差正規化,預測後再還原回 mg/dL,
           故合併訓練不會因病患間血糖水準差異而失真。

    輸出:cgm_output/lstm_pooled_vs_baseline.csv、lstm_pooled_vs_perpatient.png
    """
    _keras_setup()   # 〔稽核修正 B-07／B-16〕固定全部亂數來源;只有 ImportError 才視為缺 tensorflow
    from tensorflow.keras.callbacks import EarlyStopping

    rng = np.random.RandomState(SEED)

    # ---------- 1. 逐人準備視窗(僅用各自訓練段的統計量正規化)----------
    P = {}
    for fp in list_cgm_files():
        rid = os.path.splitext(os.path.basename(fp))[0]
        g = load_one(fp)["cgm"].values.astype("float32")
        if len(g) < look_back + 40:
            continue
        n_tr = int(len(g) * 0.8)
        mu, sd = float(g[:n_tr].mean()), float(g[:n_tr].std())
        if sd == 0:
            continue
        z = (g - mu) / sd
        Xtr, ytr = make_windows(z[:n_tr], look_back, horizon)
        Xte, yte = make_windows(z[n_tr - look_back:], look_back, horizon)
        if len(Xtr) < 30 or len(Xte) < 10:
            continue
        P[rid] = dict(pid=rid.split("_")[0], mu=mu, sd=sd, n=len(g),
                      Xtr=Xtr[..., None], ytr=ytr,
                      Xte=Xte[..., None], yte=yte)

    ids = sorted(P)
    pids = sorted({P[r]["pid"] for r in ids})
    if verbose:
        tot = sum(len(P[r]["Xtr"]) for r in ids)
        print(f"\n[LSTM-pooled] 監測 {len(ids)} 筆 / 病患 {len(pids)} 位")
        print(f"  合併訓練樣本:{tot:,} 個視窗"
              f"(逐人版平均僅 {tot // max(len(ids), 1):,} 個)")
        print(f"  驗證方式:依病患分 {n_folds} 折,每折模型完全未見該折病患")

    # ---------- 2. 依病患分折 ----------
    perm = rng.permutation(len(pids))
    fold_of_pid = {pids[j]: k % n_folds for k, j in enumerate(perm)}

    def build():
        m = build_lstm_model(look_back, horizon, units)   # 〔稽核修正 B-12〕共用架構定義
        m.compile(optimizer="adam", loss="mse")
        return m

    rows = []
    for k in range(n_folds):
        te_ids = [r for r in ids if fold_of_pid[P[r]["pid"]] == k]
        tr_ids = [r for r in ids if fold_of_pid[P[r]["pid"]] != k]
        if not te_ids or not tr_ids:
            continue
        # 訓練折內再保留約 10% 的病患作為早停用驗證集(同樣依病患分,不混同一人)
        tr_pids = sorted({P[r]["pid"] for r in tr_ids})
        n_val = max(1, len(tr_pids) // 10)
        val_pids = set(rng.permutation(tr_pids)[:n_val])
        fit_ids = [r for r in tr_ids if P[r]["pid"] not in val_pids]
        val_ids = [r for r in tr_ids if P[r]["pid"] in val_pids]

        Xf = np.concatenate([P[r]["Xtr"] for r in fit_ids])
        yf = np.concatenate([P[r]["ytr"] for r in fit_ids])
        Xv = np.concatenate([P[r]["Xtr"] for r in val_ids])
        yv = np.concatenate([P[r]["ytr"] for r in val_ids])

        model = build()
        model.fit(Xf, yf, epochs=epochs, batch_size=batch_size,
                  validation_data=(Xv, yv), verbose=0,
                  callbacks=[EarlyStopping(patience=4, restore_best_weights=True)])

        for r in te_ids:
            d = P[r]
            pred = model.predict(d["Xte"], verbose=0).ravel() * d["sd"] + d["mu"]
            act = d["yte"].ravel() * d["sd"] + d["mu"]
            base = d["Xte"][:, -1, 0] * d["sd"] + d["mu"]      # 持續性基準
            l_rmse = float(np.sqrt(np.mean((pred - act) ** 2)))
            b_rmse = float(np.sqrt(np.mean((base - act) ** 2)))
            rows.append(dict(
                record=r, patient=d["pid"], fold=k, n_points=d["n"],
                n_train_windows=len(d["Xtr"]),
                lstm_rmse=l_rmse, lstm_mae=float(np.mean(np.abs(pred - act))),
                baseline_rmse=b_rmse, baseline_mae=float(np.mean(np.abs(base - act))),
                skill_pct=(1 - l_rmse / b_rmse) * 100 if b_rmse > 0 else np.nan))
        if verbose:
            print(f"  fold {k + 1}/{n_folds} 完成(訓練 {len(fit_ids)} 筆、"
                  f"驗證 {len(val_ids)} 筆、測試 {len(te_ids)} 筆)")

    R = pd.DataFrame(rows).sort_values("record").reset_index(drop=True)
    R.to_csv(f"{OUTDIR}/lstm_pooled_vs_baseline.csv", index=False, encoding="utf-8-sig")

    win = (R.lstm_rmse < R.baseline_rmse).mean() * 100
    print(f"\n[LSTM-pooled] {len(R)} 筆完成 → {OUTDIR}/lstm_pooled_vs_baseline.csv")
    print(f"  LSTM RMSE     mean={R.lstm_rmse.mean():.2f}  median={R.lstm_rmse.median():.2f}")
    print(f"  基準 RMSE     mean={R.baseline_rmse.mean():.2f}  median={R.baseline_rmse.median():.2f}")
    print(f"  相對改善      mean={R.skill_pct.mean():.1f}%  median={R.skill_pct.median():.1f}%")
    print(f"  勝過基準比例:{win:.1f}%  ({int((R.lstm_rmse < R.baseline_rmse).sum())}/{len(R)})")

    # 依監測長度分組:檢驗合併訓練是否解決短序列過擬合
    R["_g"] = pd.qcut(R.n_points, 4, labels=["最短", "短", "長", "最長"])
    t = R.groupby("_g", observed=True).agg(
        筆數=("record", "size"), 平均點數=("n_points", "mean"),
        改善中位數=("skill_pct", "median"),
        勝率=("skill_pct", lambda x: (x > 0).mean() * 100)).round(1)
    print("\n  依監測長度分組(檢驗短序列是否仍失效):")
    print(t.to_string())

    # ---------- 3. 與逐人版對照圖 ----------
    per_fp = f"{OUTDIR}/lstm_vs_baseline_all.csv"
    if os.path.exists(per_fp):
        A = pd.read_csv(per_fp)[["record", "skill_pct"]].rename(columns={"skill_pct": "per_patient"})
        C = R[["record", "skill_pct", "n_points"]].rename(columns={"skill_pct": "pooled"})
        M = A.merge(C, on="record")
        if len(M):
            fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
            lim = [min(M.per_patient.min(), M.pooled.min()) - 5,
                   max(M.per_patient.max(), M.pooled.max()) + 5]
            ax[0].scatter(M.per_patient, M.pooled, s=28, alpha=.7,
                          color="#0d9488", edgecolor="white")
            ax[0].plot(lim, lim, "--", color="grey", lw=1)
            ax[0].axhline(0, color="#dc2626", ls=":", lw=1)
            ax[0].axvline(0, color="#dc2626", ls=":", lw=1)
            ax[0].set_xlim(lim); ax[0].set_ylim(lim)
            ax[0].set_xlabel("Per-patient model: skill vs baseline (%)")
            ax[0].set_ylabel("Pooled model: skill vs baseline (%)")
            ax[0].set_title(f"Pooled vs per-patient "
                            f"(above line = pooled better, {(M.pooled > M.per_patient).mean()*100:.0f}%)")
            M["_g"] = pd.qcut(M.n_points, 4, labels=["Q1 shortest", "Q2", "Q3", "Q4 longest"])
            gg = M.groupby("_g", observed=True)[["per_patient", "pooled"]].median()
            xx = np.arange(len(gg)); w = .36
            ax[1].bar(xx - w/2, gg.per_patient, w, color="#94a3b8", label="Per-patient")
            ax[1].bar(xx + w/2, gg.pooled, w, color="#0d9488", label="Pooled")
            ax[1].axhline(0, color="#374151", lw=1)
            ax[1].set_xticks(xx); ax[1].set_xticklabels(gg.index, fontsize=9)
            ax[1].set_ylabel("Median skill vs baseline (%)")
            ax[1].set_title("By monitoring length")
            ax[1].legend(fontsize=8.5)
            plt.tight_layout()
            plt.savefig(f"{OUTDIR}/lstm_pooled_vs_perpatient.png", dpi=150)
            _show()
            print(f"\n  對照圖 → {OUTDIR}/lstm_pooled_vs_perpatient.png")
            print(f"  合併訓練優於逐人版的比例:{(M.pooled > M.per_patient).mean()*100:.1f}%")
    return R.drop(columns=["_g"], errors="ignore")


def main():
    print("=" * 60)
    print("CGM 分析開始")
    print("=" * 60)

    met = build_metrics_table()          # 1. 指標(可併回 Summary)
    P3, L3, _ = markov_matrix(n_states=3)          # 2a. 3 態馬可夫
    markov_convergence(P3, L3, tag="_3state")      # 2c. 3 態收斂診斷
    P5, L5, _ = markov_matrix(n_states=5)          # 2b. 5 態馬可夫
    markov_convergence(P5, L5, tag="_5state")      # 2d. 5 態收斂診斷
    plot_agp()                            # 3a. AGP:全部病患合併(日內百分位)
    plot_fluctuation_grid(n=12)           # 3b. 12 位示範網格
    plot_metrics_distribution(met)        # 3c. 六項 CGM 指標分布 + 描述統計(〔稽核修正 C-03〕原寫 TIR/GMI/CV)
    plot_fluctuation(n_examples=3)        # 3d. 個別波動圖(改 None 可畫全部 109 位)
    baseline_all()                        # 5a. 全體病患的持續性基準(不需 tensorflow)
    # [修正 D] run_all.py --skip-lstm 會設 SKIP_LSTM=1,略過耗時的深度學習段落。
    #          Markov、CGM 指標、AGP、持續性基準等確定性結果仍完整產出。
    if os.environ.get("SKIP_LSTM") == "1":
        print("\n[SKIP_LSTM=1] 略過 LSTM 段落(逐人版、合併訓練版、單筆預測圖)。")
        # 〔稽核修正 B-16〕明白列出未更新之舊檔,避免誤以為是本次結果
        _old = [f for f in LSTM_OUTPUTS if os.path.exists(os.path.join(OUTDIR, f))]
        if _old:
            print(f"  ★ 以下 LSTM 檔案「未」重新產生,仍為先前執行之結果:{', '.join(_old)}")
    else:
        # 〔稽核修正 B-16〕先確認 tensorflow 可用。原寫法是在各 LSTM 函式內以 except Exception
        #   吞下任何錯誤、印一行後 return,程式仍以 0 結束,cgm_output/ 內舊的 LSTM CSV 就被
        #   當成本次結果。現改為:只有 ImportError 視為「沒有 tensorflow」→ 刪除舊 LSTM 檔、
        #   說明原因並以非 0 結束;其他例外原樣拋出(Python 亦以非 0 結束)。
        try:
            _keras_setup()
        except TensorFlowMissing as e:
            _gone = []
            for f in LSTM_OUTPUTS:
                p = os.path.join(OUTDIR, f)
                if os.path.exists(p):
                    os.remove(p); _gone.append(f)
            print(f"\n[錯誤] {e}")
            if _gone:
                print(f"  已刪除先前執行留下之 LSTM 檔(避免被誤當本次結果):{', '.join(_gone)}")
            sys.exit(2)
        # 〔稽核修正 C-02〕原本另以 lstm_forecast(patient_prefix="2000", epochs=40) 單獨訓練一個
        #   模型畫圖 4-7-5,圖上 RMSE(約 5.7)與正文取自下一行之同一病患值(5.94)出自不同次
        #   訓練。現改由 lstm_forecast_all() 以同一模型畫該圖(plot_record_prefix="2000")。
        # 5b. 全體病患 LSTM vs 基準(較耗時;可先用 n_patients=20 試跑)+ 圖 4-7-5
        lstm_forecast_all(epochs=30, n_patients=None, plot_record_prefix="2000")
        # 5c. 跨病患合併訓練 + 病患分組交叉驗證;需先跑完 5b 才能產生對照圖
        lstm_forecast_pooled(epochs=30, n_folds=5)

    print(f"\n完成。所有輸出在 ./{OUTDIR}/")
    print("圖:markov_3state.png、markov_5state.png、fluctuation_*.png、lstm_forecast_*.png")
    print("表:cgm_metrics.csv、markov_3state.csv、markov_5state.csv")


if __name__ == "__main__":
    main()