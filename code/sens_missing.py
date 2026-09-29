# -*- coding: utf-8 -*-
"""[新增分析] 高缺失率特徵之敏感度分析。
24 個建模特徵中,Ins_2h(50.5%)、Cpep_2h(41.3%)、Ins_f(30.3%) 缺失率 >=30%,
全部以中位數插補;而 Cpep_2h 在主分析中為小血管併發症的重要度第一名。
本分析比較四種特徵集,檢驗結論是否依賴這些高插補比例的變數。

〔稽核修正 A-04〕原本只跑 40 次重複(「時間預算」),而主分析(diabetes_deterioration_pipeline.py)
  為 N_REPEATS=100 次,使表 5-4-1「全部(主分析)」列與表 4-3-1 同稱主分析卻數值不同。
  現改為直接沿用主程式之 N_REPEATS(=100)與同一組 StratifiedGroupKFold(5, shuffle=True,
  random_state=SEED+r) 折分配;「全部 24 特徵(主分析)」列因此與 output/model_results.csv 之
  AUC_rep_median / AUC_rep_lo95 / AUC_rep_hi95 逐位元相同(程式結尾自動核對,不符即報錯)。
  執行時間約為原來之 2.5 倍。
〔稽核修正 A-19〕原 S.round(4) 後寫檔,報告再進位至 3 位即可能雙重進位(例:0.872491 → 0.8725 →
  0.873,正確為 0.872)。改為全精度寫出。
〔稽核修正 B-16〕原 error_score=np.nan + np.nanmean 會靜默剔除失敗折;改為計數,有失敗即報錯。
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
import numpy as np, pandas as pd, os
import diabetes_deterioration_pipeline as P
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
OUT="output"; os.makedirs(OUT,exist_ok=True)
N_REPEATS=P.N_REPEATS   # 〔稽核修正 A-04〕與主分析相同(100);原為 range(40)
d,med,targets,pid,d_raw=P.load_data()
miss=(d_raw.isna().mean()*100)
MAIN="全部 24 特徵(主分析)"
sets={
 MAIN: list(d.columns),
 "剔除缺失>=30%:留 21 個": [c for c in d.columns if miss.get(c,0)<30],
 "剔除缺失>=20%:留 13 個": [c for c in d.columns if miss.get(c,0)<20],
 "僅缺失<10%:留 10 個": [c for c in d.columns if miss.get(c,0)<10],
}
rows=[]
for name,cols in sets.items():
    for t in ["Any_Complication","Microvascular","Macrovascular"]:
        y=targets[t].astype(int)
        for kind,lab in [("lr","LR"),("gb","XGB")]:
            v=[]
            for r in range(N_REPEATS):   # 〔稽核修正 A-04〕原為 range(40)(時間預算)
                cv=StratifiedGroupKFold(5,shuffle=True,random_state=P.SEED+r)
                res=cross_validate(P.make_pipeline(kind),d[cols],y,cv=cv,groups=pid,
                                   scoring={"AUC":"roc_auc"},error_score=np.nan)
                a=np.asarray(res["test_AUC"],float)
                if np.isnan(a).any():   # 〔稽核修正 B-16〕不再以 nanmean 靜默剔除失敗折
                    raise RuntimeError(f"[{name}/{t}/{lab}] 第 {r} 次重複有 {int(np.isnan(a).sum())} 折"
                                       f"配適或計分失敗;請查明原因後重跑。")
                v.append(np.nanmean(a))
            v=np.array(v)
            rows.append(dict(feature_set=name,n_features=len(cols),target=t,model=lab,
                AUC_median=float(np.median(v)),AUC_lo95=float(np.percentile(v,2.5)),
                AUC_hi95=float(np.percentile(v,97.5))))
S=pd.DataFrame(rows)
S.to_csv(f"{OUT}/sensitivity_missing_features.csv",index=False,encoding="utf-8-sig")   # 〔稽核修正 A-19〕全精度

# 〔稽核修正 A-04〕核對:「全部 24 特徵(主分析)」列必須與 model_results.csv 之 AUC_rep_* 完全相同
_mr=os.path.join(OUT,"model_results.csv")
if os.path.exists(_mr):
    MR=pd.read_csv(_mr,float_precision="round_trip")
    _lab={"LR":"Logistic Regression (baseline)"}
    bad=[]
    for _,x in S[S.feature_set==MAIN].iterrows():
        m=MR[(MR.Target==x.target)&((MR.Model==_lab["LR"]) if x.model=="LR" else (MR.Model!=_lab["LR"]))]
        for a,b in [("AUC_median","AUC_rep_median"),("AUC_lo95","AUC_rep_lo95"),("AUC_hi95","AUC_rep_hi95")]:
            if len(m)!=1 or float(m[b].iloc[0])!=float(x[a]):
                bad.append((x.target,x.model,a,float(x[a]),None if len(m)!=1 else float(m[b].iloc[0])))
    if bad:
        print("★ 「全部 24 特徵(主分析)」列與 model_results.csv 不一致(兩者應為同一組 100×5 折):")
        for b in bad: print("   ",b)
        if any(b[1]=="LR" for b in bad):
            # LR(表 5-4-1 所列者)為確定性計算,不一致即表示兩支程式之資料或折分配不同 → 中止
            raise SystemExit(4)
        print("  (僅 XGBoost 列不一致:表示本機 XGBoost 於同一設定下兩次配適結果不同,請留意;"
              "表 5-4-1 所用之 LR 列一致。)")
    else:
        print(f"✓ 核對:「{MAIN}」6 列(LR、XGB × 3 目標)之 AUC 中位數與 95% 區間與"
              f" model_results.csv 之 AUC_rep_* 逐位元相同。")
else:
    print(f"[注意] 找不到 {_mr},未核對主分析列(請先執行 diabetes_deterioration_pipeline.py)。")

pv=S.pivot_table(index=["target","model"],columns="feature_set",values="AUC_median")
print(f"=== AUC 中位數({N_REPEATS} 次重複)===")
print(pv.round(3).to_string())
# Cpep_2h 重要度是否穩健
# 〔稽核修正 B-15〕原標題「剔除 Cpep_2h 後」不精確:比較之特徵集(缺失 >=30% 者剔除)同時剔除
#   Ins_2h、Cpep_2h、Ins_f 三項。
print("\n=== 剔除缺失 >=30% 之三項(Ins_2h、Cpep_2h、Ins_f)後小血管 AUC 變化 ===")
base=S[(S.feature_set==MAIN)&(S.target=="Microvascular")]
drop=S[(S.feature_set=="剔除缺失>=30%:留 21 個")&(S.target=="Microvascular")]
for m in ["LR","XGB"]:
    b=base[base.model==m].iloc[0]; x=drop[drop.model==m].iloc[0]
    print(f"  {m}: {b.AUC_median:.3f} [{b.AUC_lo95:.3f},{b.AUC_hi95:.3f}] → {x.AUC_median:.3f} [{x.AUC_lo95:.3f},{x.AUC_hi95:.3f}]  ({x.AUC_median-b.AUC_median:+.3f})")
