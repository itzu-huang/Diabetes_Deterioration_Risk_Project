# -*- coding: utf-8 -*-
"""
reverse_scan.py —— 第 3 層:存在性掃描
================================================================
crosscheck.py(第 2 層)查的是「具名的關鍵數字在每一處是否正確」。本腳本反過來做:
  1. 抽出四份交付物中所有「結果型數字」(附位置:報告 ¶N/表格格位、簡報第 N 張、海報文字框、App)
  2. 逐一問:本次執行的輸出檔裡,有沒有哪一格(以該數字自身之位數、ROUND_HALF_UP)算得出它?
  3. 列出「算不出來」者;算不出來且未列入 allowlist.csv 者即以非 0 結束。

〔稽核修正 B-03〕結論改稱「存在性掃描」:本層只證明「這個數字可以在某個輸出中找到」,
  不證明它出現的位置、語境與所指之量正確(那是第 2 層的工作)。稽核實測:把報告之成本倍數
  2.98 全改為 2.89,原版本層反把 2.89 列為「強證據」(因 LDL 原始值與某個重要度恰為 2.89)。
  本版之「強證據」須同時符合:
    (a) 命中來源格數 ≤ --max-sources;
    (b) 命中來源為「分析輸出」之欄——原始資料(Shanghai_T2DM_Summary.xlsx)、
        diagnostics_v2.csv 中 source_type 為 literature/code_constant 之列(舊版無此欄時,
        以 vandoorn_*、cost_direct_*、cost_total_*、cost_ratio_both_*、cpi_cum_*、
        lstm_params_approx 推定)不算;
    (c) 命中來源與數字之「語境」相符(例:前後文談成本者,來源須為成本相關輸出);
    (d) 數字本身至少 3 位有效數字(0.4、21.0 這類過短之數字巧合機率太高)。
  命中但語境不符者另列「▲ 語境不符之巧合」,只供人工檢視,不當作任何證據。
〔稽核修正 CODE-29〕allowlist.csv 之豁免須符合「前後文」(正規式,須出現在該數字前後 40 字
  或該格之表題/列/欄中),不再以(檔,值)一體豁免整份檔案。
〔稽核修正 DOCS-17〕交付物與輸出目錄改以標記檔自動搜尋,或以 --base/--out 指定;
  掃描輸出時略過 reference*/(含 reference_old_*/)。

[修正 28] 之設計(依 token 自身位數、單一進位方式比對;涵蓋率自我揭露;掃描 App 常數區;
剔除 data: URI)保留。

用法:
    python reverse_scan.py
    python reverse_scan.py --base <交付物目錄> --out <程式與輸出目錄>
    python reverse_scan.py --max-sources 3     # 命中來源 <= N 格者才可能是強證據
結束碼:0＝全部數字皆可在輸出(或豁免清單)中找到;1＝有找不到出處之數字。
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
import decimal
import glob
import math
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = decimal.Decimal

# ======================================================================
# 〔稽核修正 DOCS-17〕目錄搜尋:以標記檔辨識
# ======================================================================
_SKIP = {"__pycache__", ".git", "Shanghai_T2DM", "figures_v2", "reference", "reference_teacher",
         "eda_figures", "cgm_output", "output", "output_reg", "output_risk", "output_cost_mc"}


def _is_deliv(d):
    return bool(glob.glob(os.path.join(d, "專題書面報告_v*.docx")))


def _is_out(d):
    return os.path.isfile(os.path.join(d, "output_risk", "quadrant_summary.csv"))


def _find(pred, explicit=""):
    if explicit:
        p = os.path.abspath(explicit)
        return p if pred(p) else ""
    d = HERE
    for _ in range(5):
        frontier, budget = [(d, 0)], 4000
        while frontier:
            x, dep = frontier.pop(0)
            budget -= 1
            if budget < 0:
                break
            if pred(x):
                return os.path.abspath(x)
            if dep >= 4:
                continue
            try:
                subs = sorted(e.path for e in os.scandir(x) if e.is_dir() and e.name not in _SKIP
                              and not e.name.startswith("reference_old_"))
            except OSError:
                continue
            frontier.extend((s, dep + 1) for s in subs)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return ""


# [修正 31] 交付物檔名帶版本號,取版本號最大者。
def pick(base, pattern):
    cs = sorted(glob.glob(os.path.join(base, pattern)))
    if not cs:
        return ""

    def ver(p):
        m = re.search(r"_v(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0
    return max(cs, key=ver)


def deliverables(base):
    d = {"報告": pick(base, "專題書面報告_v*.docx"),
         "海報": pick(base, "專題海報_80x100cm_v*.pptx"),
         "簡報": pick(base, "專題簡報_v*.pptx")}
    appdir = pick(base, "糖三臟決策支援App_v*")
    d["App"] = os.path.join(appdir, "index.html") if appdir else ""
    d["App2"] = os.path.join(appdir, "糖三臟決策支援App.html") if appdir else ""
    return d


# [修正 28f] 只把「真正的減號」正規化為 ASCII 連字號;破折號與連接號換成空白。
_TRANS = str.maketrans({"−": "-", "－": "-", "％": "%", "，": ",", "．": ".",
                        **{chr(0xFF10 + i): str(i) for i in range(10)}})
_DASHES = str.maketrans({"–": " ", "—": " ", "‐": " ", "‑": " "})


def norm(t):
    return (t or "").translate(_TRANS).translate(_DASHES)


# ======================================================================
# 文字單位:(交付物, 位置, 文字, 語境)
# ======================================================================
def units_docx(p):
    import docx
    from docx.oxml.ns import qn
    d = docx.Document(p)
    U = [("報告", f"¶{i}", x.text, "") for i, x in enumerate(d.paragraphs) if x.text.strip()]
    body = list(d.element.body.iterchildren())
    caps = []
    for i, ch in enumerate(body):
        if ch.tag != qn("w:tbl"):
            continue
        cap = ""
        for j in range(i - 1, max(i - 4, -1), -1):
            if body[j].tag == qn("w:p"):
                t = "".join(x.text or "" for x in body[j].iter(qn("w:t"))).strip()
                if t.startswith("表 "):
                    cap = t
                    break
                if t:
                    break
        caps.append(cap)
    for ti, t in enumerate(d.tables):
        cap = caps[ti] if ti < len(caps) else ""
        tag = (cap.split("　")[0].strip() if cap else "表") + f"(T{ti})"
        rows = []
        for r in t.rows:
            seen, cells = set(), []
            for c in r.cells:
                if id(c._tc) not in seen:
                    seen.add(id(c._tc))
                    cells.append(c.text.strip())
            rows.append(cells)
        hdr = rows[0] if rows else []
        for ri, cells in enumerate(rows):
            for ci, tx in enumerate(cells):
                if tx:
                    U.append(("報告", f"{tag} r{ri}c{ci}", tx,
                              f"{cap}｜{cells[0] if cells else ''}｜{hdr[ci] if ci < len(hdr) else ''}"))
    return U


def units_pptx(p, dv):
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    def flat(shapes):
        for sh in shapes:
            if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
                yield from flat(sh.shapes)
            else:
                yield sh
    U = []
    for si, s in enumerate(Presentation(p).slides, 1):
        shs = list(flat(s.shapes))
        texts = [sh.text_frame.text if sh.has_text_frame else "" for sh in shs]
        where = f"第{si}張 " if dv == "簡報" else ""
        for k, sh in enumerate(shs):
            if sh.has_text_frame and texts[k].strip():
                nb = " ".join(t for t in texts[max(0, k - 1):k + 2] if t)
                U.append((dv, f"{where}{'本文' if dv == '簡報' else '文字框'}〈{sh.name}〉", texts[k], nb))
            if getattr(sh, "has_table", False) and sh.has_table:
                rows = [[c.text.strip() for c in r.cells] for r in sh.table.rows]
                hdr = rows[0] if rows else []
                for ri, cells in enumerate(rows):
                    for ci, tx in enumerate(cells):
                        if tx:
                            U.append((dv, f"{where}表 r{ri}c{ci}", tx,
                                      f"{cells[0] if cells else ''}｜{hdr[ci] if ci < len(hdr) else ''}"))
        if s.has_notes_slide:
            t = s.notes_slide.notes_text_frame.text
            if t.strip():
                U.append((dv, f"{where}備註", t, ""))
    return U


def _const_blocks(s):
    out = {}
    for m in re.finditer(r"\bconst\s+([A-Za-z_$][\w$]*)\s*=\s*([\{\[])", s):
        name, op = m.group(1), m.group(2)
        cl = "}" if op == "{" else "]"
        j, d = m.end() - 1, 0
        for k in range(j, len(s)):
            if s[k] == op:
                d += 1
            elif s[k] == cl:
                d -= 1
                if d == 0:
                    out.setdefault(name, s[j:k + 1])
                    break
    return out


def units_html(p):
    """[修正 28d] 可見文字與 const 區塊原文分開;[修正 28h] 先剔除 base64。"""
    s = open(p, encoding="utf-8", errors="replace").read()
    s = re.sub(r"data:[a-zA-Z0-9.+/-]+;base64,[A-Za-z0-9+/=\s]+", "data:...", s)
    s = re.sub(r"[A-Za-z0-9+/]{120,}={0,2}", "<base64>", s)
    U = [("App-常數區", f"const {k}", v, "") for k, v in sorted(_const_blocks(s).items())]
    body = re.sub(r"<script.*?</script>", " ", s, flags=re.S)
    body = re.sub(r"<style.*?</style>", " ", body, flags=re.S)
    U.append(("App", "可見文字", re.sub(r"<[^>]+>", " ", body), ""))
    return U


# ======================================================================
# 輸出之數值格:(值, 來源, 類別)——類別:output / raw / literature / code_constant
# ======================================================================
SCALES = [(1.0, ""), (100.0, "×100"), (0.01, "÷100"), (1e-4, "÷10000")]
_LEGACY_LIT = re.compile(r"^(vandoorn_|cost_direct_|cost_total_)")
_LEGACY_CONST = re.compile(r"^(cost_ratio_both_|cpi_cum_|lstm_params_approx)")


def load_cells(out_dir, teacher=False):
    cells = []
    for f in sorted(glob.glob(os.path.join(out_dir, "**", "*.csv"), recursive=True)):
        rel = os.path.relpath(f, out_dir).replace(os.sep, "/")
        top = rel.split("/")[0]
        if teacher:
            if top != "reference_teacher":
                continue
        elif top.startswith("reference"):              # 〔稽核修正 DOCS-17〕含 reference_old_*
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        kinds = None
        if os.path.basename(rel) == "diagnostics_v2.csv" and "item" in df.columns:
            if "source_type" in df.columns:
                kinds = [("literature" if st == "literature" else "code_constant" if st == "code_constant"
                          else "output") for st in df["source_type"].astype(str)]
            else:
                kinds = [("literature" if _LEGACY_LIT.search(str(it)) else
                          "code_constant" if _LEGACY_CONST.search(str(it)) else "output")
                         for it in df["item"].astype(str)]
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            for i, v in enumerate(df[c].values):
                try:
                    fv = float(v)
                except Exception:
                    continue
                if math.isfinite(fv):
                    cells.append((fv, f"{rel}::{c}", kinds[i] if kinds else "output"))
    return cells


def load_raw_cells(out_dir):
    """[修正 28i] 原始資料集之數值亦為合法出處(App 示範病患),但不得作為強證據。"""
    cells = []
    for f in glob.glob(os.path.join(out_dir, "Shanghai_T2DM_Summary.xls*")):
        try:
            df = pd.read_excel(f, sheet_name="T2DM")
        except Exception:
            continue
        for c in df.columns:
            for x in pd.to_numeric(df[c], errors="coerce").dropna().values:
                if math.isfinite(float(x)):
                    cells.append((float(x), f"Shanghai_T2DM_Summary.xlsx::{c}", "raw"))
    return cells


def build_index(cells, precisions):
    """{(精度, 字串): {(來源, 類別)}} —— 只用 ROUND_HALF_UP。[修正 28a]"""
    idx = {}
    for fv, src, kind in cells:
        for sc, _lab in SCALES:
            x = fv * sc
            if not math.isfinite(x) or abs(x) > 1e12:
                continue
            dx = D(repr(x))
            for nd in precisions:
                q = D(1).scaleb(-nd)
                try:
                    s = str(dx.quantize(q, rounding=decimal.ROUND_HALF_UP))
                except decimal.InvalidOperation:
                    continue
                idx.setdefault((nd, s), set()).add((src, kind))
    return idx


PAT = [
    (r"(?<![\d.])-?0\.\d{3,4}(?![\d])", "小數比值(3–4 位)"),
    (r"(?<![\d.])\d{1,3}\.\d%", "百分比(1 位小數)"),
    (r"(?<![\d.])\d{1,3}\.\d{2}%", "百分比(2 位小數)"),
    (r"(?<![\d.])\d{1,3}\.\d\s*萬", "萬元金額(1 位小數)"),
    (r"(?<![\d.])-?\d{1,4}\.\d{2}(?![\d%])", "兩位小數(RMSE/MAE 等)"),
    (r"(?<![\d,])\d{1,3}(?:,\d{3})+(?![\d,])", "千分位金額/計數"),
    (r"n\s?=\s?\d{1,4}", "n = 樣本數"),
    (r"(?<![\d.])\d{1,3}/\d{1,3}(?![\d])", "比例(分數式)"),
]

IGNORE = {
    "2001", "2003", "2011", "2015", "2016", "2017", "2018", "2019", "2020",
    "2021", "2022", "2023", "2024", "2025", "2026",
    "70", "180", "54", "250", "36", "15", "14", "12", "5", "3", "4", "10",
    "20", "30", "50", "100", "109", "110", "1000",
    "0.5", "1.0", "2.0", "0.0", "3.0", "4.0", "5.0", "0.05", "0.95",
    "0.025", "0.975",
}

# 〔稽核修正 B-03〕語境 → 相符之來源(前後文關鍵詞, 來源正規式)
TOPICS = [
    (r"AUC|曲線下面積|判別|\bAP\b|Precision|Recall|\bF1\b|召回|精確率|交叉驗證結果",
     r"model_results|sensitivity_missing|epv_vs_performance|cost_sensitivity.*_auc|diagnostics_v2"
     r"|collinearity_ablation"),                        # 〔稽核修正 B-10〕剔除特徵之 AUC(¶803)
    # 〔稽核修正 B-10〕共線性(¶521、¶574、¶585、¶802):VIF、判定係數、拔靴、兩兩相關
    (r"VIF|變異數膨脹|共線|拔靴|剔除|判定係數|定義式|最大偏差|相關係數|相關|\br\s*[=≈]",
     r"collinearity_|eda_figures/correlation|diagnostics_v2"),
    (r"RMSE|MAE|均方根|改善|技巧分數|勝率|勝過|LSTM|預測誤差|視窗|時界|持續性基準|基準\s*RMSE",
     r"cgm_output/lstm_|cgm_output/baseline_all|output/lstm_by_length_group|diagnostics_v2"),
    (r"TIR|TAR|TBR|範圍內|低於範圍|高於範圍|變異係數|GMI|平均血糖|監測|血糖",
     r"cgm_output/cgm_|cgm_output/markov_|patient_risk_table|diagnostics_v2"),
    (r"轉移|穩態|λ|特徵值|半衰期|馬可夫|收斂|自／至|自＼至", r"cgm_output/markov_"),
    (r"象限|併發症比例|長期風險|短期風險|切點|門檻|分層", r"output_risk/|quadrant_cost_interval|diagnostics_v2"),
    (r"成本|費用|萬元|萬|元|Gini|吉尼|百分位|倍|支出|母體|金額", r"output_cost_mc/|diagnostics_v2"),
    (r"勝算|\bOR\b|↑|↓", r"lr_oddsratio|diagnostics_v2"),
    (r"R²|決定係數|MAPE|迴歸|糖化血色素", r"output_reg/|lstm_pooled_by_horizon|diagnostics_v2"),
    (r"重要|SHAP", r"importance"),
    (r"缺失|偏態|平均數|標準差|中位數|最小|最大|敘述|第一四分位", r"eda_figures/|cgm_descriptive|cgm_metrics|diagnostics_v2"),
    (r"分群|群\s*\d|silhouette|BIC|K-means|高斯混合", r"cluster_"),
    (r"EPV|每變數事件", r"epv_vs_performance"),
    (r"CPI|物價|通膨|(?<!質量)指數|累計倍數|環比", r"diagnostics_v2|life_table"),   # 〔稽核修正 B-10〕排除「身體質量指數」
    (r"餘命|生命表|年金|存活|歲", r"life_table_ex|patient_lifetime_cost_mc"),
    (r"盛行|陽性|預測均值|平均預測|實際", r"cost_sensitivity_class_weight|diagnostics_v2|output_risk/"),
    (r"特徵", r"sensitivity_missing|epv_vs_performance|importance"),
]
TOPICS = [(re.compile(a, re.I), re.compile(b)) for a, b in TOPICS]


def tok_precision(num):
    return len(num.split(".")[1]) if "." in num else 0


def sig_digits(num):
    return len(re.sub(r"[^\d]", "", num).lstrip("0"))


def load_allow(path, tool):
    rows = []
    if not os.path.exists(path):
        return rows
    import io
    raw = open(path, "rb").read()
    try:
        txt = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # 〔稽核修正 CODE-29〕Windows 上以 Excel「CSV(逗號分隔)」另存會變成 cp950;改以 cp950 讀取並提示
        txt = raw.decode("cp950", errors="replace")
        print(f"[allow] {os.path.basename(path)} 不是 UTF-8(可能由 Excel 另存為「CSV(逗號分隔)」),"
              f"已改以 cp950 讀取;建議另存為「CSV UTF-8(逗號分隔)」。")
    # 「#」開頭為註解(Excel 另存後可能變成 "# …" 帶引號,一併視為註解)
    txt = "\n".join(l for l in txt.splitlines() if not l.lstrip().lstrip('"').startswith("#"))
    df = pd.read_csv(io.StringIO(txt), dtype=str).fillna("")
    for _, r in df.iterrows():
        use = (r.get("適用", "") or "*").strip()
        if use not in ("*", tool):
            continue
        ctx = (r.get("前後文", "") or "").strip()
        if not ctx:
            print(f"[allow] 略過未寫前後文之豁免列:{r.get('檔')},{r.get('值')}(豁免必須指明前後文)")
            continue
        try:
            cx = re.compile(ctx)
        except re.error as e:                          # 〔稽核修正 CODE-29〕前後文寫錯時略過該列並指出原因
            print(f"[allow] 略過前後文不是有效正規式之豁免列:{r.get('檔')},{r.get('值')},{ctx!r}({e})")
            continue
        rows.append((r["檔"].strip(), r["值"].strip(), cx, r.get("理由", "").strip()))
    return rows


def allowed(rows, dv, tok, window, ctx):
    for f, v, cx, why in rows:
        if v == tok and f in ("*", dv) and (cx.search(window) or cx.search(ctx)):
            return why
    return None


def main():
    ap = argparse.ArgumentParser(description="第 3 層:存在性掃描")
    ap.add_argument("--base", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-sources", type=int, default=3,
                    help="命中來源 <= 此值者,才可能列為強證據")
    ap.add_argument("--allow", default="", help="豁免清單,預設讀同目錄之 allowlist.csv")
    a = ap.parse_args()

    BASE = _find(_is_deliv, a.base)
    OUT = _find(_is_out, a.out)
    if not BASE or not OUT:
        sys.exit(f"[錯誤] 找不到交付物或輸出目錄(BASE={BASE!r} OUT={OUT!r}),請用 --base/--out 指定。")
    print("【第 3 層】存在性掃描(只證明數字可在某個輸出中找到,不證明其位置與語境正確)")
    print(f"[path] 交付物:{BASE}")
    print(f"[path] 程式輸出:{OUT}")
    DV = deliverables(BASE)
    for k, v in DV.items():
        print(f"[file] {k}: {os.path.basename(v) if v else '(找不到)'}")
    U = units_docx(DV["報告"]) + units_pptx(DV["海報"], "海報") + units_pptx(DV["簡報"], "簡報") + units_html(DV["App"])

    cells = load_cells(OUT)
    raw = load_raw_cells(OUT)
    if raw:
        print(f"[scope] 另納入原始資料集 {len(raw):,} 格(僅作為 App 示範病患之出處;不作為強證據)")
    ref_cells = load_cells(OUT, teacher=True)
    print(f"[scope] 本次執行之數值格:{len(cells):,}"
          f"(其中文獻值/程式常數 {sum(1 for c in cells if c[2] != 'output'):,});"
          f"reference_teacher/(另一次執行):{len(ref_cells):,}")
    PRECS = (0, 1, 2, 3, 4)
    idx = build_index(cells + raw, PRECS)
    ridx = build_index(ref_cells, PRECS) if ref_cells else {}

    def cov(nd, lo, hi, step):
        tot = hit = 0
        x = lo
        while x <= hi + 1e-12:
            tot += 1
            if (nd, f"{x:.{nd}f}") in idx:
                hit += 1
            x += step
        return hit, tot
    h1, t1 = cov(1, 0.0, 100.0, 0.1)
    h3, t3 = cov(3, 0.0, 0.999, 0.001)
    h0, t0 = cov(0, 0, 100, 1)
    print(f"[鑑別力] 比對集涵蓋率:0.0–100.0 一位小數 {h1}/{t1}({h1/t1*100:.1f}%)、"
          f"0.000–0.999 三位小數 {h3}/{t3}({h3/t3*100:.1f}%)、0–100 整數 {h0}/{t0}({h0/t0*100:.1f}%)")
    print("          涵蓋率愈高,「找得到」愈不具鑑別力——隨手寫一個一位小數之百分比,")
    print(f"          約有 {h1/t1*100:.0f}% 機率可在某個輸出中找到。故本層只能找出「完全找不到出處」的數字,")
    print("          不能證明數字正確;逐處正確性由第 2 層 crosscheck.py 負責。")

    allow_path = a.allow or os.path.join(HERE, "allowlist.csv")
    ALLOW = load_allow(allow_path, "reverse_scan")
    print(f"[allow] 豁免清單 {os.path.basename(allow_path)}:{len(ALLOW)} 列(須符合前後文才豁免)")

    strong = weak = vague = 0
    allowed_list, unresolved, ref_only, strong_rows, mismatch_ctx, only_nonout = [], [], [], [], [], []
    for dv, loc, raw_t, ctx in U:
        t = norm(raw_t)
        c2 = norm(ctx)
        seen = set()
        for pat, kind in PAT:
            for m in re.finditer(pat, t):
                tok = m.group(0)
                num = re.sub(r"[^\d.\-]", "", tok)
                if num in IGNORE or num in ("", "-", ".") or (loc, tok) in seen:
                    continue
                seen.add((loc, tok))
                win = t[max(0, m.start() - 30):m.end() + 15]
                snip = raw_t[max(0, m.start() - 20):m.end() + 20].replace("\n", " ")
                if re.fullmatch(r"\d{1,3}/\d{1,3}", tok):                # 分數式:分子分母各查
                    aa, bb = tok.split("/")
                    if idx.get((tok_precision(aa), aa)) and idx.get((tok_precision(bb), bb)):
                        vague += 1
                        continue
                    why = allowed(ALLOW, dv, tok, win, c2)
                    if why:
                        allowed_list.append((dv, loc, tok, why))
                    else:
                        unresolved.append((dv, loc, kind, tok, snip))
                    continue
                nd = tok_precision(num)
                hits = idx.get((nd, num)) or idx.get((nd, num.lstrip("0") or num))
                if not hits:
                    why = allowed(ALLOW, dv, tok, win, c2)
                    if why:
                        allowed_list.append((dv, loc, tok, why))
                    elif ridx.get((nd, num)):
                        ref_only.append((dv, loc, kind, tok, snip))
                    else:
                        unresolved.append((dv, loc, kind, tok, snip))
                    continue
                srcs = {s for s, _k in hits}
                outs = {s for s, k in hits if k == "output"}
                topics = [src_rx for ctx_rx, src_rx in TOPICS if ctx_rx.search(win) or ctx_rx.search(c2)]
                ok_ctx = [s for s in outs if any(rx.search(s) for rx in topics)]
                if not outs:
                    only_nonout.append((dv, loc, tok, sorted(f"{s}〔{k}〕" for s, k in hits)[:3], snip))
                elif topics and not ok_ctx:
                    mismatch_ctx.append((dv, loc, tok, sorted(srcs)[:3], snip))
                if len(srcs) <= a.max_sources and ok_ctx and sig_digits(num) >= 3:
                    strong += 1
                    strong_rows.append((dv, loc, tok, sorted(ok_ctx)[:3]))
                elif len(srcs) <= 20:
                    weak += 1
                else:
                    vague += 1

    tot = strong + weak + vague + len(unresolved) + len(ref_only) + len(allowed_list)
    print(f"\n{'=' * 110}")
    print(f"共掃出 {tot} 個結果型數字 token(依位置分別計)")
    print(f"  可找到且語境相符、來源 ≤ {a.max_sources} 格之分析輸出(強):{strong}")
    print(f"  可找到但來源較多或僅為原始資料/文獻值(弱):{weak}")
    print(f"  可找到但來源 >20 格(無鑑別力):{vague}")
    print(f"    ▲ 上兩類中,只見於原始資料/文獻值/程式常數者(不算證據):{len(only_nonout)}")
    print(f"    ▲ 上兩類中,在分析輸出中找得到但語境不符者(數值巧合,不算證據):{len(mismatch_ctx)}")
    print(f"  ★ 本次執行找不到出處:{len(unresolved)}")
    print(f"  ★ 本次執行找不到、但 reference_teacher(另一次執行)找得到:{len(ref_only)}")
    print(f"  ○ 依豁免清單(前後文相符)豁免:{len(allowed_list)}")
    if allowed_list:
        print("\n○ 已豁免之數字及其理由:")
        for dv, loc, tok, why in allowed_list:
            print(f"   [{dv}] {loc} {tok:<10} {why[:90]}")
    if ref_only:
        print(f"\n★ 只有另一次執行找得到者({len(ref_only)} 筆)——高度可疑,請逐條核對:")
        for dv, loc, kind, tok, snip in ref_only:
            print(f"   [{dv}] {loc} {tok:<10} …{snip}…")
    if unresolved:
        print(f"\n★ 找不到出處者({len(unresolved)} 筆)——可能是文獻、流行病學統計或門檻值,但每一個都應有出處"
              "(若確認無誤,請於 allowlist.csv 寫明「前後文」與理由):")
        for dv, loc, kind, tok, snip in unresolved:
            print(f"   [{dv}] {loc} [{kind}] {tok:<10} …{snip}…")
    if only_nonout:
        print(f"\n▲ 只見於原始資料/文獻值/程式常數({len(only_nonout)} 筆;這些數字在本層不算被覆核):")
        for dv, loc, tok, srcs, snip in only_nonout[:60]:
            print(f"   [{dv}] {loc} {tok:<10} ← {', '.join(srcs)}  …{snip}…")
        if len(only_nonout) > 60:
            print(f"   …另有 {len(only_nonout) - 60} 筆")
    if mismatch_ctx:
        print(f"\n▲ 語境不符之巧合({len(mismatch_ctx)} 筆,僅供人工檢視;若其中有應出自分析輸出之數字,"
              "請以第 2 層核對):")
        for dv, loc, tok, srcs, snip in mismatch_ctx[:80]:
            print(f"   [{dv}] {loc} {tok:<10} ← {', '.join(s for s in srcs)}  …{snip}…")
        if len(mismatch_ctx) > 80:
            print(f"   …另有 {len(mismatch_ctx) - 80} 筆")
    if strong_rows:
        print(f"\n{'=' * 110}")
        print("強證據明細(語境相符之分析輸出、來源 ≤ "
              f"{a.max_sources} 格;注意:仍只是「存在」,不代表位置正確)")
        for dv, loc, tok, ss in strong_rows[:150]:
            print(f"  [{dv}] {loc} {tok:<10} ← {', '.join(ss)}")
        if len(strong_rows) > 150:
            print(f"  …另有 {len(strong_rows) - 150} 筆")
    if unresolved or ref_only:
        print("\n結論:未通過 —— 有數字在本次輸出中找不到出處(見上)。")
        sys.exit(1)
    print("\n結論:存在性掃描通過 —— 每個結果型數字都能在本次輸出(或附前後文之豁免清單)中找到出處;"
          "此結論不代表數字之位置與語境正確。")
    sys.exit(0)


if __name__ == "__main__":
    main()
