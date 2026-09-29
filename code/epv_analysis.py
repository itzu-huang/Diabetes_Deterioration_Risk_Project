# -*- coding: utf-8 -*-
"""[新增分析] 每變數事件數(EPV)與效能／不確定性的關係。
Collins 等(BMC Med 2011;9:103)將 EPV<10 列為第 2 型糖尿病風險預測模型之
方法學缺陷。本研究之 EPV 為 1.13~1.79,遠低於該標準。
本分析以「依缺失率由低到高逐步納入特徵」建構不同 EPV 的模型,
檢驗 EPV 提高後 AUC 與其 95% 區間寬度的變化。
〔稽核修正 B-16〕交叉驗證失敗折改為計數並報錯(原以 np.nanmean 靜默剔除);輸出不變。
(本檔重複 R=60 次、各欄寫檔前 round 至 4 位,皆維持原設計:報告僅於表 3-7-1 列出本檔,
 未引用本檔之 AUC 數值;¶819 之 AUC 與區間寬度出自 sens_missing.py。另注意 EPV 欄以 Python
 round() 取 2 位,27/24=1.125 會成 1.12(銀行家進位),報告之 1.13 為四捨五入,兩者皆非錯誤。)"""

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
import numpy as np, pandas as pd, os
import diabetes_deterioration_pipeline as P
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
R = 60
d, med, targets, pid, d_raw = P.load_data()
miss = (d_raw.isna().mean()*100)
order = sorted(d.columns, key=lambda c: (miss.get(c, 0), c))   # 缺失率低者優先
rows = []
for k in [6, 9, 13, 16, 20, 24]:
    cols = order[:k]
    for t in ["Any_Complication", "Microvascular", "Macrovascular"]:
        y = targets[t].astype(int); ev = int(y.sum())
        for kind, lab in [("lr", "LR"), ("gb", "XGB")]:
            v = []
            for r in range(R):
                cv = StratifiedGroupKFold(5, shuffle=True, random_state=P.SEED+r)
                a = np.asarray(cross_validate(P.make_pipeline(kind), d[cols], y,
                    cv=cv, groups=pid, scoring={"AUC": "roc_auc"}, error_score=np.nan)["test_AUC"], float)
                # 〔稽核修正 B-16〕原 error_score=np.nan + np.nanmean 會靜默剔除失敗折;改為有失敗即報錯
                #   (無失敗時 np.nanmean 與原寫法完全相同,輸出不變)。
                if np.isnan(a).any():
                    raise RuntimeError(f"[k={k}/{t}/{lab}] 第 {r} 次重複有 {int(np.isnan(a).sum())} 折配適或"
                                       f"計分失敗;請查明原因後重跑。")
                v.append(np.nanmean(a))
            v = np.array(v)
            lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
            rows.append(dict(n_features=k, EPV=round(ev/k, 2), target=t, model=lab,
                AUC_median=round(float(np.median(v)), 4),
                AUC_lo95=round(float(lo), 4), AUC_hi95=round(float(hi), 4),
                CI_width=round(float(hi-lo), 4),
                max_missing_pct=round(float(max(miss.get(c, 0) for c in cols)), 1)))
S = pd.DataFrame(rows)
S.to_csv("output/epv_vs_performance.csv", index=False, encoding="utf-8-sig")
print("=== EPV 與 AUC 中位數 ===")
print(S.pivot_table(index=["target","model"], columns="EPV", values="AUC_median").round(3).to_string())
print("\n=== EPV 與 95% 區間寬度(越窄越穩定)===")
print(S.pivot_table(index=["target","model"], columns="EPV", values="CI_width").round(3).to_string())
print("\n=== 特徵數 / EPV / 最大缺失率 對照 ===")
print(S[S.target=="Any_Complication"][S.model=="LR"][["n_features","EPV","max_missing_pct"]].to_string(index=False))
