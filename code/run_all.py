# -*- coding: utf-8 -*-
"""
run_all.py —— 從原始資料一鍵重現全部結果
================================================================
用途:消除原流程唯一的人工步驟(手動解壓 CGM 壓縮檔),並固定全部分析程式的
執行順序,使任何人在任何機器上都能以單一指令重跑整個專題。
〔稽核修正 B-15〕原寫「固定五支程式的執行順序」,現行預設為下列八支;加 --with-sensitivity
  另跑五支,共十三支(見 SENSITIVITY)。

執行方式:
    python run_all.py --clean --with-sensitivity   # 建議:先清除舊輸出,再完整重跑全部 13 支
    python run_all.py                  # 重跑預設 8 支(不清除舊輸出;不含敏感度/診斷/共線性)
    python run_all.py --skip-lstm      # 略過耗時的 LSTM 訓練(cgm_lstm_markov.py 之 LSTM 段落
                                       #   與 pooled_ph.py);LSTM 相關結果維持舊檔,不可宣稱全流程重現
    python run_all.py --check-only     # 只驗資料指紋與環境,不執行分析(不改寫 run_log.json)
    〔稽核修正 B-09〕--clean:執行前刪除各輸出目錄(見 OUTPUT_DIRS)內由程式產生之 *.csv、*.png
    與 pooled_ph.py 之指紋檔 *.meta.json,保留說明檔(*.txt),並列出刪了哪些檔;reference/ 與
    reference_teacher/ 不受影響。--clean 時 pooled_ph.py 一律以 --force 全部重算。
    --clean 不可與 --skip-lstm 併用(LSTM 輸出會被刪除卻不重新產生)。

執行順序(有相依關係,不可調換):
    1. diabetes_deterioration_pipeline.py  → output/         (共用 load_data)
    2. diabetes_regression.py              → output_reg/     (import 上者)
    3. cgm_lstm_markov.py                  → cgm_output/     (獨立)
    4. pooled_ph.py                        → cgm_output/     (需 3;PH=30/60 分)
    5. 產生時界彙總檔.py                    → cgm_output/     (需 3 與 4)
    6. deterioration_risk.py               → output_risk/    (需 1 與 CGM 資料)
    7. lifetime_cost_montecarlo.py         → output_cost_mc/ (需 1 與 6)
    8. make_figures_v2.py                  → figures_v2/     (需前七支之輸出)
    [修正 22:4、5、8 原本不在清單內,「一鍵重現」跑完仍缺三項被引用之結果。]
  --with-sensitivity(前八支全部成功後才執行):
    9. sens_missing.py                     → output/         (高缺失特徵敏感度;表 5-4-1)
   10. epv_analysis.py                     → output/         (EPV 與效能)
   11. make_diagnostics.py                 → output/         (diagnostics_v2.csv、lstm_by_length_group.csv)
   12. collinearity_check.py               → output/         (〔稽核修正 B-10〕VIF、拔靴、剔除特徵 AUC)
   13. sensitivity_extra.py                → output/         (〔稽核修正 A-14/A-15/A-21/A-26〕共病、對數轉換、
                                                              成本參數之敏感度與一階馬可夫檢查)

產出的稽核檔:
    data_manifest.json   原始資料每一檔的 SHA-256(證明資料未被改動)
    env_lock.json        實際執行時的直譯器與套件版本
    run_log.json         每支程式的執行時間、結束碼、產出檔案清單、
                         實際命令列(argv)、以及該支程式「新增/改動了哪些檔」
                         [修正 23]
    〔稽核修正 B-08〕run_log.json 每次執行一律重寫(每跑完一支即更新,中途失敗或中斷亦保留
      已完成之步驟,status 欄標示 running/completed/failed/interrupted),並加入:Python 完整
      版本與平台、各套件版本、原始資料指紋(檔數與合併 SHA-256)、每支程式本身之 SHA-256、
      本次新增/改動之每個輸出檔之 SHA-256、--clean 所刪除之檔案清單。輸出目錄之快照改為
      遞迴(含 output/eda_figures/)。
    〔稽核修正 B-16〕任一程式失敗時,run_all.py 本身以結束碼 1 結束(原為印「有失敗」後
      仍以 0 結束);--with-sensitivity 所列之程式不存在時亦報錯(原為靜默略過)。
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
import fnmatch
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
CGM_DIR = os.path.join(HERE, "Shanghai_T2DM")
SUMMARY = os.path.join(HERE, "Shanghai_T2DM_Summary.xlsx")

# [修正 22] 原清單只有 5 支,但交付物中另有三項結果只能由以下程式產生,
#   不列入即「一鍵重現」跑完後仍有數個被引用的 CSV/圖檔不存在:
#     pooled_ph.py          → cgm_output/lstm_pooled_PH30m.csv、lstm_pooled_PH60m.csv
#                             (PH=30/60 分之合併訓練 LSTM 逐筆結果)
#     產生時界彙總檔.py      → cgm_output/lstm_pooled_by_horizon.csv(上兩檔與 PH=15 之彙總;
#                             報告表 5-4-2、5-4-3 之 30/60 分鐘數字出自此檔)
#     make_figures_v2.py    → 交付物實際採用之圖檔
#   pooled_ph.py 需在 cgm_lstm_markov.py 之後(共用同一套切分),
#   產生時界彙總檔.py 需在 pooled_ph.py 之後。
#   〔稽核修正 B-15〕原寫舊檔名 _pooled_ph30m.csv、_pooled_ph60m.csv([修正 25a] 已改名),
#   並稱其為「報告表 4-3-2、簡報第 15 張之 PH=30/60 分結果」(報告無表 4-3-2,簡報第 15 張
#   為二維風險分層;見稽核 C-07),已改正。
SCRIPTS = [
    ("diabetes_deterioration_pipeline.py", "output"),
    ("diabetes_regression.py", "output_reg"),
    ("cgm_lstm_markov.py", "cgm_output"),
    ("pooled_ph.py", "cgm_output"),
    ("產生時界彙總檔.py", "cgm_output"),
    ("deterioration_risk.py", "output_risk"),
    ("lifetime_cost_montecarlo.py", "output_cost_mc"),
    ("make_figures_v2.py", "figures_v2"),   # [修正 32] 原寫 output,以致誤報「未寫出任何檔案」
]
# --with-sensitivity 另跑者(〔稽核修正 B-10〕加入 collinearity_check.py)
SENSITIVITY = [
    ("sens_missing.py", "output"),
    ("epv_analysis.py", "output"),
    ("make_diagnostics.py", "output"),
    ("collinearity_check.py", "output"),
    ("sensitivity_extra.py", "output"),     # 〔稽核修正 A-14/A-15/A-21/A-26〕補做之敏感度與假設檢查
]
# 〔稽核修正 B-09〕--clean 之清除範圍:以下目錄內(不遞迴)由程式產生之檔案
OUTPUT_DIRS = ["output", "output/eda_figures", "output_reg", "output_risk",
               "output_cost_mc", "cgm_output", "figures_v2"]
CLEAN_PATTERNS = ["*.csv", "*.png", "*.meta.json"]

PACKAGES = ["pandas", "numpy", "scipy", "sklearn", "matplotlib", "seaborn",
            "openpyxl", "xlrd", "xgboost", "shap", "tensorflow", "keras"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_cgm_extracted():
    """[修正 D] 原流程要求使用者自行把 CGM 壓縮檔解壓成同層 Shanghai_T2DM/,
    該步驟只寫在 requirements.txt 的註解裡、沒有任何程式執行,是全流程唯一
    無法自動重現的環節。此處自動完成。"""
    if os.path.isdir(CGM_DIR) and len(os.listdir(CGM_DIR)) > 0:
        print(f"[data] {os.path.basename(CGM_DIR)}/ 已存在({len(os.listdir(CGM_DIR))} 檔),略過解壓")
        return
    zips = [f for f in os.listdir(HERE)
            if f.startswith("Shanghai_T2DM") and f.endswith(".zip")]
    if not zips:
        sys.exit("[錯誤] 找不到 Shanghai_T2DM*.zip,也沒有 Shanghai_T2DM/ 資料夾。")
    zp = os.path.join(HERE, sorted(zips)[0])
    print(f"[data] 自動解壓 {os.path.basename(zp)} …")
    with zipfile.ZipFile(zp) as z:
        z.extractall(HERE)
    print(f"[data] 完成,共 {len(os.listdir(CGM_DIR))} 檔")


def write_data_manifest():
    """原始資料指紋。任何一筆 CGM 或摘要檔被改動,重跑時即可立刻發現。"""
    files = {}
    if os.path.exists(SUMMARY):
        files[os.path.basename(SUMMARY)] = sha256(SUMMARY)
    # 〔稽核修正 A-25〕只計入資料檔(*.xls/*.xlsx)。資料夾內另附之授權說明 DATA_LICENSE.txt 等文件
    #   不屬原始資料,不列入指紋,故合併指紋仍為原 110 檔之值(c186c2db…)。
    for fn in sorted(os.listdir(CGM_DIR)):
        if not fn.lower().endswith((".xls", ".xlsx")):
            continue
        files[f"Shanghai_T2DM/{fn}"] = sha256(os.path.join(CGM_DIR, fn))
    mani = {"n_files": len(files),
            "combined_sha256": hashlib.sha256(
                "".join(f"{k}:{v}" for k, v in sorted(files.items())).encode()).hexdigest(),
            "files": files}
    out = os.path.join(HERE, "data_manifest.json")
    prev = None
    if os.path.exists(out):
        try:
            prev = json.load(open(out, encoding="utf-8"))
        except Exception:
            prev = None
    json.dump(mani, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[data] 指紋:{len(files)} 檔,合併 SHA-256 = {mani['combined_sha256'][:16]}…")
    if prev and prev.get("combined_sha256") != mani["combined_sha256"]:
        print("[警告] 原始資料指紋與上次執行不同!請確認資料是否被改動。")
    return mani


def write_env_lock():
    env = {"python": sys.version.split()[0],
           "platform": platform.platform(),
           "machine": platform.machine(),
           "packages": {}}
    for p in PACKAGES:
        try:
            m = __import__(p)
            env["packages"][p] = getattr(m, "__version__", "unknown")
        except Exception:
            env["packages"][p] = "NOT INSTALLED"
    json.dump(env, open(os.path.join(HERE, "env_lock.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[env] Python {env['python']} / {env['platform']}")
    miss = [k for k, v in env["packages"].items() if v == "NOT INSTALLED"]
    if miss:
        print(f"[env] 缺少套件:{', '.join(miss)}  → pip install -r requirements.txt")
    return env


def clean_outputs():
    """〔稽核修正 B-09〕刪除各輸出目錄內由程式產生之檔案(*.csv、*.png、*.meta.json),
    保留其餘檔案(例如 figures_v2/_這個資料夾是什麼.txt);不動 reference/、reference_teacher/。
    回傳 (已刪除清單, 保留清單),路徑以 / 分隔、相對於本資料夾。"""
    removed, kept = [], []
    for d in OUTPUT_DIRS:
        p = os.path.join(HERE, *d.split("/"))
        if not os.path.isdir(p):
            continue
        for fn in sorted(os.listdir(p)):
            fp = os.path.join(p, fn)
            if not os.path.isfile(fp):
                continue
            if any(fnmatch.fnmatch(fn.lower(), pat) for pat in CLEAN_PATTERNS):
                os.remove(fp)
                removed.append(f"{d}/{fn}")
            else:
                kept.append(f"{d}/{fn}")
    print(f"[clean] 已刪除 {len(removed)} 個由程式產生之舊輸出:")
    for r in removed:
        print(f"         - {r}")
    if kept:
        print(f"[clean] 保留(非程式產生之說明檔等):{', '.join(kept)}")
    return removed, kept


def _snapshot(outdir):
    """輸出目錄之 (相對路徑 → mtime_ns, size) 快照。
    〔稽核修正 B-08〕原本不遞迴,output/eda_figures/ 內檔案之新增/改動不會被記錄;改為遞迴。"""
    p = os.path.join(HERE, outdir)
    if not os.path.isdir(p):
        return {}
    out = {}
    for root, dirs, files in os.walk(p):
        dirs.sort()
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            out[os.path.relpath(fp, p).replace(os.sep, "/")] = (st.st_mtime_ns, st.st_size)
    return out


def run_one(script, outdir, extra_args=()):
    """[修正 23] 原本只記錄「執行後該目錄有幾個檔」,無法分辨這些檔是本次
    產生的、還是上一次執行留下的。若某支程式其實沒有寫出任何檔(或寫到別的
    目錄),n_outputs 照樣是個大數字,看起來一切正常。改為執行前後各取一次
    快照,明確記錄本次「新增」與「改動」了哪些檔;另記錄實際命令列 argv,
    使 --skip-lstm 這類會改變結果的旗標留在 run_log.json 裡。
    〔稽核修正 B-08〕另記錄程式本身之 SHA-256 與本次新增/改動之每個輸出檔之 SHA-256。"""
    print(f"\n{'='*64}\n[run] {script}  → {outdir}/\n{'='*64}")
    spath = os.path.join(HERE, script)
    if not os.path.exists(spath):   # 〔稽核修正 B-16〕不再靜默略過
        print(f"[錯誤] 找不到程式 {script}")
        return {"script": script, "outdir": outdir, "returncode": -1, "argv": [],
                "seconds": 0.0, "error": "script not found", "script_sha256": None,
                "n_outputs": 0, "outputs": [], "created": [], "modified": [],
                "n_created": 0, "n_modified": 0, "output_sha256": {}}
    before = _snapshot(outdir)
    cmd = [sys.executable, script] + list(extra_args)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE)
    dt = time.time() - t0
    after = _snapshot(outdir)
    produced = sorted(after)
    created = sorted(set(after) - set(before))
    modified = sorted(k for k in set(after) & set(before) if after[k] != before[k])
    status = "OK" if r.returncode == 0 else f"FAILED(rc={r.returncode})"
    print(f"[run] {script} {status}  {dt:.1f}s  "
          f"目錄內 {len(produced)} 檔(本次新增 {len(created)}、改動 {len(modified)})")
    if r.returncode == 0 and not created and not modified:
        print(f"[警告] {script} 結束碼為 0,但 {outdir}/ 沒有任何檔案被新增或改動 —— "
              f"請確認它是否真的產生了輸出。")
    out_sha = {}
    for rel in created + modified:
        fp = os.path.join(HERE, outdir, *rel.split("/"))
        if os.path.isfile(fp):
            out_sha[rel] = sha256(fp)
    return {"script": script, "outdir": outdir, "returncode": r.returncode,
            "argv": [os.path.basename(sys.executable)] + cmd[1:],
            "seconds": round(dt, 1), "script_sha256": sha256(spath),
            "n_outputs": len(produced), "outputs": produced,
            "created": created, "modified": modified,
            "n_created": len(created), "n_modified": len(modified),
            "output_sha256": out_sha}


def _write_log(log):
    """〔稽核修正 B-08〕每跑完一支即重寫 run_log.json(中途失敗或中斷亦留下紀錄)。"""
    log["total_seconds"] = round(sum(s["seconds"] for s in log["steps"]), 1)
    json.dump(log, open(os.path.join(HERE, "run_log.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-lstm", action="store_true",
                    help="略過 LSTM 訓練(cgm_lstm_markov.py 之 LSTM 段落與 pooled_ph.py;"
                         "仍會跑 Markov 與 CGM 指標)")
    ap.add_argument("--check-only", action="store_true", help="只驗資料與環境(不改寫 run_log.json)")
    ap.add_argument("--with-sensitivity", action="store_true",
                    help="加跑敏感度分析、診斷輸出與共線性診斷(sens_missing、epv_analysis、"
                         "make_diagnostics、collinearity_check、sensitivity_extra)")
    ap.add_argument("--clean", action="store_true",               # 〔稽核修正 B-09〕
                    help="執行前刪除各輸出目錄內由程式產生之 CSV/PNG(保留說明 txt),"
                         "並令 pooled_ph.py 全部重算")
    a = ap.parse_args()
    if a.clean and a.skip_lstm:
        ap.error("--clean 不可與 --skip-lstm 併用:--clean 會刪除 LSTM 輸出,--skip-lstm 卻不重新產生,"
                 "下游程式將因缺檔而失敗。")

    print("=" * 64)
    print("糖尿病惡化風險專題 —— 一鍵重現")
    print("=" * 64)
    ensure_cgm_extracted()
    mani = write_data_manifest()
    env = write_env_lock()
    if a.check_only:
        print("\n[check-only] 資料與環境檢查完成,未執行分析(run_log.json 未改寫)。")
        return

    if a.skip_lstm:
        os.environ["SKIP_LSTM"] = "1"
        print("[opt] --skip-lstm:cgm_lstm_markov.py 之 LSTM 段落與 pooled_ph.py 將略過")

    log = {"run_log_format": 2,                                  # 〔稽核修正 B-08〕
           "status": "running",
           "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "finished": None,
           "runner_argv": sys.argv,                    # [修正 23] 記錄本次旗標
           "skip_lstm": bool(a.skip_lstm),
           "with_sensitivity": bool(a.with_sensitivity),
           "clean": bool(a.clean),
           "python_full": sys.version,
           "platform": platform.platform(),
           "machine": platform.machine(),
           "data_manifest": {"n_files": mani["n_files"],
                             "combined_sha256": mani["combined_sha256"]},
           "data_manifest_sha": mani["combined_sha256"], "env": env,
           "cleaned_files": [], "steps": []}
    if a.clean:
        log["cleaned_files"], _ = clean_outputs()
    _write_log(log)

    plan = list(SCRIPTS)
    ok = True
    try:
        for script, outdir in plan:
            extra = ["--force"] if (a.clean and script == "pooled_ph.py") else []
            log["steps"].append(run_one(script, outdir, extra))
            _write_log(log)
            if log["steps"][-1]["returncode"] != 0:
                print(f"\n[中止] {script} 執行失敗,後續程式有相依關係,停止。")
                ok = False
                break
        if ok and a.with_sensitivity:
            for script, outdir in SENSITIVITY:
                log["steps"].append(run_one(script, outdir))
                _write_log(log)
                if log["steps"][-1]["returncode"] != 0:
                    print(f"\n[中止] {script} 執行失敗,停止。")
                    ok = False
                    break
        log["status"] = "completed" if ok else "failed"
    except KeyboardInterrupt:
        log["status"] = "interrupted"
        ok = False
        print("\n[中斷] 使用者中斷;run_log.json 已記錄至中斷前完成之步驟。")
    finally:
        log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_log(log)

    print(f"\n{'='*64}")
    ok = ok and all(s["returncode"] == 0 for s in log["steps"])
    _silent = [s["script"] for s in log["steps"]
               if s["returncode"] == 0 and not s["n_created"] and not s["n_modified"]]
    print(f"全部完成:{'成功' if ok else '有失敗'}  總時間 {log['total_seconds']:.0f}s "
          f"({len(log['steps'])} 支程式)")
    if _silent:                                        # [修正 23]
        print(f"★ 以下程式結束碼為 0 但未寫出任何檔案:{', '.join(_silent)}")
    if a.skip_lstm:
        print("★ 本次使用 --skip-lstm,LSTM 相關結果並未重新產生,"
              "不可據以宣稱「全流程重現」。")
    if not a.with_sensitivity:
        print("★ 本次未加 --with-sensitivity:sensitivity_missing_features、epv_vs_performance、"
              "diagnostics_v2、lstm_by_length_group 與 collinearity_* 未重新產生。")
    print("稽核檔:data_manifest.json、env_lock.json、run_log.json")
    print("下一步:python verify_outputs.py  (比對本次輸出與參考結果)")
    if not ok:
        sys.exit(1)   # 〔稽核修正 B-16〕原本有失敗仍以 0 結束


if __name__ == "__main__":
    main()
