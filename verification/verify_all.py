# -*- coding: utf-8 -*-
"""
verify_all.py —— 一鍵執行四層驗證
================================================================
  第 1 層 與凍結快照比對   verify_outputs.py  本次輸出 vs reference/(另一次完整執行之凍結結果);
                                              第一段印出快照出處;快照後未重跑即判「自我比對」失敗
  第 2 層 交付物數字逐處比對 crosscheck.py    報告/簡報/海報/App 中每一處出現之關鍵數字 vs 本次輸出
  第 3 層 存在性掃描       reverse_scan.py    交付物之數字能否在本次輸出中找到(只證明「存在」,
                                              不證明「位置正確」)
  第 4 層 四捨五入核對     rounding_check.py  交付物表格格位 vs 全精度值(ROUND_HALF_UP)

〔稽核修正 B-18〕(1) 只選部分層(--layer)時,未執行之層在結果表標示 SKIPPED,結論固定寫
  「僅執行第 N 層;未執行:…;本結果不代表四層通過」;只有四層都實際執行且結束碼皆為 0
  (且第 1 層不是 --allow-stale 之形式檢查)才印「全部四層皆實際執行且通過」。
  (2) 本檔在重現包之「程式與輸出」目錄與交付物之「驗證腳本」目錄各放一份,兩份逐位元
  相同;各層腳本與資料一律以本檔位置(__file__)向上/向下搜尋「標記檔」找到,或以
  --stage-dir、--deliverables-dir、--scripts-dir 指定。
〔稽核修正 DOCS-17〕不再寫死「【學生版】程式與輸出★」等候選路徑;資料夾改名後仍可找到。
〔稽核修正 C-10〕各層名稱如實:第 1 層是「與凍結快照比對」(不是「程式輸出可重現」之證明,
  除非快照後確有另一次完整重跑);第 3 層是「存在性掃描」(不是「每個數字可覆核」)。
〔稽核修正 B-18〕驗證報告檔名含時間戳(verify_report_<年月日_時分秒>.txt),並寫入每層之
  結束碼與其意義、實際使用之目錄、Python 版本與命令列。

[修正 27] 之設計(以結束碼判定、完整保存各層輸出、逐層結果表)保留。

用法:
    python verify_all.py                          # 四層全跑(同平台)
    python verify_all.py --cross-platform         # 第 1 層以跨平台模式比對(另一作業系統之重跑)
    python verify_all.py --layer 1                # 只跑第 1 層(結論會註明其餘未執行)
    python verify_all.py --layer 2 3 4            # 只跑第 2–4 層
    python verify_all.py --stage-dir <程式與輸出目錄> --deliverables-dir <交付物目錄>
    python verify_all.py --allow-stale            # 第 1 層僅做形式檢查(非可重現性證據)
結束碼:0＝所選各層皆實際執行且通過;1＝有層未通過或因找不到腳本而未執行。
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
import glob
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# 〔稽核修正 C-10〕(層號, 名稱, 腳本, 所在, 性質說明)
LAYERS = [
    (1, "與凍結快照比對", "verify_outputs.py", "stage",
     "本次輸出 vs reference/ 快照;快照後未重跑即判失敗"),
    (2, "交付物數字逐處比對", "crosscheck.py", "scripts",
     "交付物中每一處關鍵數字 vs 本次輸出"),
    (3, "存在性掃描", "reverse_scan.py", "scripts",
     "數字能否在輸出中找到;不證明位置正確"),
    (4, "四捨五入核對", "rounding_check.py", "scripts",
     "表格格位 vs 全精度值"),
]
RC_MEANING = {
    1: {0: "通過", 1: "未通過", 2: "無從比對(無快照或無輸出)", 3: "自我比對——快照後未重跑"},
    2: {0: "通過", 1: "有不符或無法檢查之項目"},
    3: {0: "通過", 1: "有找不到出處之數字"},
    4: {0: "通過(邊界警告如有,見報告)", 1: "有不符或找不到表格"},
}

# 〔稽核修正 DOCS-17〕以標記檔辨識目錄,不寫死資料夾名稱
SKIP_DIRS = {"__pycache__", ".git", "node_modules", "Shanghai_T2DM", "figures_v2",
             "reference", "reference_teacher", "eda_figures", "cgm_output", "output",
             "output_reg", "output_risk", "output_cost_mc", "media"}


def is_stage(d):
    return (os.path.isfile(os.path.join(d, "verify_outputs.py"))
            and os.path.isfile(os.path.join(d, "run_all.py")))


def is_deliv(d):
    return bool(glob.glob(os.path.join(d, "專題書面報告_v*.docx")))


def is_scripts(d):
    return all(os.path.isfile(os.path.join(d, f))
               for f in ("crosscheck.py", "reverse_scan.py", "rounding_check.py"))


def _walk_down(root, pred, max_depth, budget):
    """由 root 往下以廣度優先搜尋第一個符合 pred 之目錄(略過輸出與資料夾)。"""
    frontier = [(root, 0)]
    while frontier:
        d, dep = frontier.pop(0)
        budget[0] -= 1
        if budget[0] < 0:
            return ""
        if pred(d):
            return d
        if dep >= max_depth:
            continue
        try:
            subs = sorted(e.path for e in os.scandir(d)
                          if e.is_dir() and e.name not in SKIP_DIRS
                          and not e.name.startswith("reference_old_"))
        except OSError:
            continue
        frontier.extend((s, dep + 1) for s in subs)
    return ""


def find_dir(pred, start, max_up=4, max_depth=4):
    """自 start 起,逐層往上;每一層往下搜尋。回傳最近之符合目錄。"""
    d = os.path.abspath(start)
    seen = set()
    for _ in range(max_up + 1):
        hit = _walk_down(d, lambda x: x not in seen and pred(x), max_depth, [4000])
        if hit:
            return os.path.abspath(hit)
        seen.add(d)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return ""


def resolve(explicit, pred, what):
    if explicit:
        p = os.path.abspath(explicit)
        if not pred(p):
            return "", f"指定之{what} {p} 不符(缺少標記檔)"
        return p, "命令列指定"
    p = find_dir(pred, HERE)
    return p, ("依本檔位置自動搜尋" if p else "找不到")


def parse_layers(vals):
    out = []
    for v in vals or []:
        for x in str(v).replace(",", " ").split():
            if x.strip():
                out.append(int(x))
    bad = [x for x in out if x not in (1, 2, 3, 4)]
    if bad:
        sys.exit(f"[錯誤] --layer 只能是 1–4:{bad}")
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description="一鍵執行四層驗證")
    ap.add_argument("--layer", nargs="*", default=[], help="只跑某幾層,例如 --layer 1 或 --layer 2 3 4")
    ap.add_argument("--stage-dir", default="", help="「程式與輸出」目錄(含 run_all.py、verify_outputs.py、輸出)")
    ap.add_argument("--deliverables-dir", default="", help="交付物目錄(含 專題書面報告_v*.docx 等)")
    ap.add_argument("--scripts-dir", default="", help="第 2–4 層腳本所在目錄(預設自動搜尋)")
    ap.add_argument("--cross-platform", action="store_true", help="第 1 層以跨平台模式比對")
    ap.add_argument("--allow-stale", action="store_true",
                    help="第 1 層在快照後未重跑時仍比對(僅形式檢查,非可重現性證據)")
    ap.add_argument("--allow-skip", action="store_true",
                    help="因找不到腳本而未執行之層不使結束碼為 1(僅供除錯;結論仍會註明)")
    ap.add_argument("--report", default="", help="報告檔路徑;預設為本檔目錄之 verify_report_<時間>.txt")
    a = ap.parse_args()

    want = parse_layers(a.layer) or [1, 2, 3, 4]
    stage, how_s = resolve(a.stage_dir, is_stage, "「程式與輸出」目錄")
    deliv, how_d = resolve(a.deliverables_dir, is_deliv, "交付物目錄")
    if a.scripts_dir:
        scripts, how_c = resolve(a.scripts_dir, is_scripts, "腳本目錄")
    elif is_scripts(HERE):
        scripts, how_c = HERE, "本檔所在目錄"
    elif deliv and is_scripts(os.path.join(deliv, "驗證腳本")):
        scripts, how_c = os.path.join(deliv, "驗證腳本"), "交付物目錄下之 驗證腳本/"
    else:
        scripts, how_c = resolve("", is_scripts, "腳本目錄")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    rpt_path = a.report or os.path.join(HERE, f"verify_report_{stamp}.txt")
    mm_path = os.path.splitext(rpt_path)[0] + "_crosscheck_mismatch.csv"
    buf = []

    def say(s=""):
        print(s)
        buf.append(s)

    say("=" * 78)
    say(f"四層驗證　{time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"命令列:{' '.join([os.path.basename(sys.argv[0])] + sys.argv[1:])}")
    say(f"Python:{sys.version.split()[0]}({platform.platform()})")
    say(f"本檔位置:{HERE}")
    say(f"「程式與輸出」目錄(第 1 層):{stage or '(找不到)'}  ← {how_s}")
    say(f"交付物目錄(第 2–4 層):{deliv or '(找不到)'}  ← {how_d}")
    say(f"第 2–4 層腳本目錄:{scripts or '(找不到)'}  ← {how_c}")
    say(f"選擇之層:{'、'.join(map(str, want))}"
        + ("　(第 1 層:跨平台模式)" if a.cross_platform else "")
        + ("　(第 1 層:--allow-stale 形式檢查)" if a.allow_stale else ""))
    say("=" * 78)

    results = []
    _env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")   # [修正 32]
    for n, name, script, where, desc in LAYERS:
        if n not in want:
            results.append((n, name, "SKIPPED(未選)", None, None))    # 〔稽核修正 B-18〕
            continue
        say(f"\n{'=' * 78}\n第 {n} 層　{name}　({script};{desc})\n{'=' * 78}")
        base = stage if where == "stage" else scripts
        path = os.path.join(base, script) if base else ""
        need = []
        if not path or not os.path.exists(path):
            need.append(f"找不到 {script}")
        if where == "scripts" and not deliv:
            need.append("找不到交付物目錄(可用 --deliverables-dir 指定)")
        if where == "scripts" and not stage:
            need.append("找不到「程式與輸出」目錄(可用 --stage-dir 指定)")
        if need:
            say("  ★ 未執行:" + ";".join(need))
            results.append((n, name, "SKIPPED(找不到)", None, None))
            continue
        if where == "stage":
            cmd = [sys.executable, path, "--stage-dir", stage]
            if a.cross_platform:
                cmd.append("--cross-platform")
            if a.allow_stale:
                cmd.append("--allow-stale")
            cwd = stage
        else:
            cmd = [sys.executable, path, "--base", deliv, "--out", stage]
            if script == "crosscheck.py":                 # 〔稽核修正 B-03〕工作清單與報告同名並存
                cmd += ["--mismatch-csv", mm_path]
            cwd = scripts
        say("  指令:" + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        t0 = time.time()
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=_env)
        dt = time.time() - t0
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if (r.stderr or "").strip() else "")
        buf.append(out)                                  # [修正 27c] 完整保存
        lines = [l for l in out.strip().split("\n") if l.strip()]
        show = lines[:8] + [l for l in lines if ("★" in l or "▲" in l)][:30] + lines[-4:]
        seen, uniq = set(), []
        for l in show:
            if l not in seen:
                seen.add(l)
                uniq.append(l)
        print("\n".join("  " + (l if len(l) <= 220 else l[:217] + "…") for l in uniq))
        if "ModuleNotFoundError" in (r.stderr or ""):
            say("  ★ 缺少套件;第 2–4 層需要:pip install python-docx python-pptx pandas numpy")
        meaning = RC_MEANING.get(n, {}).get(r.returncode, "未預期之結束碼")
        st = "PASS" if r.returncode == 0 else "FAIL"
        say(f"  → 第 {n} 層 {st}(exit {r.returncode}:{meaning})　{dt:.1f} 秒　(完整輸出見報告檔)")
        results.append((n, name, st, r.returncode, dt))

    say("\n" + "=" * 78)
    say(f"{'層':<3}{'名稱':<14}{'結果':<16}{'exit':>6}{'秒':>9}　意義")      # [修正 27e]
    say("-" * 78)
    for n, name, st, rc, dt in results:
        meaning = RC_MEANING.get(n, {}).get(rc, "") if rc is not None else "—"
        say(f"{n:<3}{name:<14}{st:<16}{('—' if rc is None else str(rc)):>6}"
            f"{('—' if dt is None else f'{dt:.1f}'):>9}　{meaning}")
    say("-" * 78)

    ran = [x for x in results if x[3] is not None]
    fails = [x for x in ran if x[2] != "PASS"]
    notfound = [x for x in results if x[2] == "SKIPPED(找不到)"]
    unselected = [x for x in results if x[2] == "SKIPPED(未選)"]
    ok = (not fails) and (not notfound or a.allow_skip)
    ran_ns = "、".join(str(x[0]) for x in ran) or "無"

    # 〔稽核修正 B-18〕結論措辭:只有四層都實際執行且通過,才可宣稱四層通過。
    partial = bool(unselected or notfound)
    if notfound:
        say("★ 有 " + str(len(notfound)) + " 層因找不到腳本或目錄而未執行:"
            + "、".join(f"第 {x[0]} 層" for x in notfound) + "。未執行不是通過。")
    if fails:
        head = "結論:未通過 —— " + ";".join(
            f"第 {x[0]} 層(exit {x[3]}:{RC_MEANING.get(x[0], {}).get(x[3], '')})" for x in fails) + "。"
    elif not ran:
        head = "結論:沒有任何一層實際執行。"
    elif partial:
        head = f"結論:所執行之第 {ran_ns} 層結束碼皆為 0。"
    elif a.allow_stale:
        head = ("結論:四層皆已執行且結束碼為 0;但第 1 層是 --allow-stale 之形式檢查"
                "(快照後未重跑),不構成可重現性證據,故不得宣稱四層通過。")
    elif a.cross_platform:
        head = ("結論:全部四層皆實際執行且通過(第 1 層為跨平台模式:確定性結果全數相符,"
                "XGBoost/LSTM 相依之差異僅列警告,見報告檔)。")
    else:
        head = "結論:全部四層皆實際執行且通過。"
    say(head)
    if partial:
        skipped = "、".join(f"第 {x[0]} 層" for x in sorted(unselected + notfound))
        say(f"      僅執行第 {ran_ns} 層;未執行:{skipped};本結果不代表四層通過。")
    if fails:
        say(f"      各層完整輸出見報告檔 {os.path.basename(rpt_path)}。")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(buf) + "\n")
    print(f"\n完整報告:{rpt_path}")
    if 2 in want and os.path.exists(mm_path):
        print(f"第 2 層工作清單(需更新之數字位置):{mm_path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
