# -*- coding: utf-8 -*-
"""[新增分析] 跨病患合併訓練 LSTM 在文獻標準預測時界(PH)之表現。
原程式僅做 next-step(PH=15 分),該時界持續性基準最強、且與 CGM 預測文獻
慣用之 PH=30/60 分不可比。本檔在同一套資料切分下加跑 PH=30、60 分。

[修正 25] 三處修補:
 (a) 輸出檔名。原寫 cgm_output/_pooled_ph{N}m.csv,但變更說明_v2.md 第 66 行
     與參考快照 reference/ 內皆為 cgm_output/lstm_pooled_PH{N}m.csv。兩者內容
     相同、僅檔名不同,於是 verify_outputs.py 兩邊都對不上(參考側那兩個檔
     在原版的比對邏輯中根本不會被走訪,故從未被發現)。本版統一採用文件所載
     之 lstm_pooled_PH{N}m.csv,並自動沿用既有之舊檔名資料(若存在)。
     〔稽核修正 CODE-21〕原寫「第 60 行」,實為第 66 行。
 (b) 必須帶兩個位置參數才能執行(STEP 與 BUDGET),無預設值。run_all.py 以
     `python pooled_ph.py` 呼叫時會直接 IndexError。本版改為 argparse:
     不帶參數時依序跑 PH=30(STEP=2)與 PH=60(STEP=4),各自獨立子行程。
     〔稽核修正 B-07〕原稱此作法「與原本 pooled_ph.py 2 / pooled_ph.py 4 逐位元等價
     (分開行程才能確保 RandomState 與 TF 種子狀態一致)」並不成立:原程式只呼叫
     tf.random.set_seed,而 Keras 3 之權重初始化種子取自 Python random(從未設定),
     兩次執行之初始權重必然不同。現改為每一折開始時以 cgm_lstm_markov._keras_setup
     (SEED+折號) 設定 random / NumPy / TF / Keras 全部種子並開啟 op determinism,
     同一台機器上重跑逐位元相同(實測見 _測試紀錄/CODE/B-07_種子/)。
 (c) 時間預算 BUDGET 用完時會靜默跳過剩餘折,程式仍以 0 結束、輸出只有部分
     折的資料,下游彙總照算不誤。本版在折數不足時明確警告並以非 0 結束碼結束。

〔稽核修正 B-09〕續跑(已完成之折不重算)之有效性檢查:
  原本只要 CSV 內「出現過」某折就視為完成,且完整性只檢查「折數 = 5」:
  上游 LSTM 重跑、資料或程式改過、或上次在寫檔中途被中斷(只寫入部分列),
  都會被當成有效結果沿用,使三個時界混用不同次訓練。現改為:
  (1) 於 cgm_output/lstm_pooled_PH{N}m.meta.json 記錄「設定 + 資料指紋 + 程式指紋
      + 套件版本」;與本次不符(或沒有此檔,例如舊版程式之產物)即刪除舊結果、全部重算;
  (2) 指紋相符時,逐折比對「該折應有之紀錄清單」(依病患分折之結果),
      筆數或紀錄不符之折捨棄重算;
  (3) 全部完成後再逐折檢查一次,並將列排成固定順序(折號、紀錄),使「中斷後續跑」與
      「一次跑完」之輸出檔相同;
  (4) --force:不論現況,刪除既有結果全部重算(run_all.py --clean 亦會先刪除)。
〔稽核修正 B-16〕沒有 tensorflow(ImportError)時:刪除本時界之舊結果、說明原因並以
  非 0 結束;設定 SKIP_LSTM=1(run_all.py --skip-lstm)時明確略過並列出未更新之舊檔。

用法:
  python pooled_ph.py              # 依序跑 PH=30、60(各自獨立子行程)
  python pooled_ph.py 2            # 只跑 PH=30(STEP=2)
  python pooled_ph.py 4 3600       # 只跑 PH=60,時間預算 3600 秒(超過即不再開新折)
  python pooled_ph.py --force      # 刪除既有結果,全部重算
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
import argparse,hashlib,json,os,platform,subprocess,sys,time
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import numpy as np, pandas as pd, cgm_lstm_markov as C
# ---- [修正 25b] 參數解析 ----------------------------------------------
_ap=argparse.ArgumentParser(description="跨病患合併訓練 LSTM 之多預測時界評估")
_ap.add_argument("step",nargs="?",type=int,default=None,
                 help="預測步數(1 步 = 15 分):2 = PH30、4 = PH60。不給則依序跑 2 與 4")
_ap.add_argument("budget",nargs="?",type=float,default=1e9,
                 help="時間預算(秒);超過即停止開新折。預設不限")
_ap.add_argument("--force",action="store_true",                      # 〔稽核修正 B-09〕
                 help="刪除既有之 lstm_pooled_PH{N}m.csv 與其指紋檔,全部重算")
_A=_ap.parse_args()
if _A.step is None:
    # 不帶參數:依序以子行程跑 STEP=2、4,確保每次都是全新的亂數與 TF 狀態
    _rc=0
    for _st in (2,4):
        print(f"\n[pooled_ph] === STEP={_st}(PH={_st*15} 分)===",flush=True)
        _r=subprocess.run([sys.executable,os.path.abspath(__file__),str(_st),str(_A.budget)]
                          +(["--force"] if _A.force else []),
                          cwd=os.getcwd())
        _rc=_rc or _r.returncode
    sys.exit(_rc)
STEP=_A.step; BUDGET=_A.budget
look_back,epochs,n_folds,bs,units=12,30,5,256,64
OUT=f"cgm_output/lstm_pooled_PH{STEP*15}m.csv"          # [修正 25a]
META=f"cgm_output/lstm_pooled_PH{STEP*15}m.meta.json"   # 〔稽核修正 B-09〕設定＋資料指紋
_LEGACY=f"cgm_output/_pooled_ph{STEP*15}m.csv"          # 舊檔名

# ---- 〔稽核修正 B-16〕略過與缺 tensorflow 之處理 ------------------------------
if os.environ.get("SKIP_LSTM")=="1":
    print(f"[SKIP_LSTM=1] 略過 PH={STEP*15} 分之 LSTM 訓練。")
    if os.path.exists(OUT):
        print(f"  ★ {OUT} 未重新產生,仍為先前執行之結果。")
    sys.exit(0)
try:
    C._keras_setup()
except C.TensorFlowMissing as _e:
    _gone=[p for p in (OUT,META) if os.path.exists(p)]
    for _p in _gone: os.remove(_p)
    print(f"[錯誤] {_e}")
    if _gone: print(f"  已刪除先前執行留下之檔案(避免被誤當本次結果):{', '.join(_gone)}")
    sys.exit(2)
from tensorflow.keras.callbacks import EarlyStopping

if os.path.exists(_LEGACY) and not os.path.exists(OUT):
    os.replace(_LEGACY,OUT)
    print(f"[pooled_ph] 既有之舊檔名 {_LEGACY} 已更名為 {OUT}(內容未變)")
rng=np.random.RandomState(C.SEED)
P={}
for fp in C.list_cgm_files():
    rid=os.path.splitext(os.path.basename(fp))[0]
    g=C.load_one(fp)["cgm"].values.astype("float32")
    if len(g)<look_back+40+STEP: continue
    n_tr=int(len(g)*0.8); mu,sd=float(g[:n_tr].mean()),float(g[:n_tr].std())
    if sd==0: continue
    z=(g-mu)/sd
    Xtr,ytr=C.make_windows(z[:n_tr],look_back,STEP)
    Xte,yte=C.make_windows(z[n_tr-look_back:],look_back,STEP)
    if len(Xtr)<30 or len(Xte)<10: continue
    P[rid]=dict(pid=rid.split("_")[0],mu=mu,sd=sd,n=len(g),
                Xtr=Xtr[...,None],ytr=ytr[:,STEP-1:STEP],
                Xte=Xte[...,None],yte=yte[:,STEP-1:STEP])
ids=sorted(P); pids=sorted({P[r]["pid"] for r in ids})
perm=rng.permutation(len(pids)); fold_of={pids[j]:k%n_folds for k,j in enumerate(perm)}
# 〔稽核修正 B-09〕每折「應有」之測試紀錄(依病患分折;與下方訓練迴圈之 te 相同)
EXPECTED={k:[r for r in ids if fold_of[P[r]["pid"]]==k] for k in range(n_folds)}


# ---- 〔稽核修正 B-09〕設定＋資料＋程式指紋 -------------------------------------
def _sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def _fingerprint():
    import tensorflow as tf, keras
    files=C.list_cgm_files()
    data=hashlib.sha256("".join(f"{os.path.basename(f)}:{_sha(f)}" for f in files).encode()).hexdigest()
    return {
        "設定":{"STEP":STEP,"PH_min":STEP*15,"look_back":look_back,"epochs":epochs,
                "n_folds":n_folds,"batch_size":bs,"units":units,"SEED":C.SEED,
                "每折種子":"cgm_lstm_markov._keras_setup(SEED+折號)",
                "早停驗證病患":"RandomState(SEED+100+折號)"},
        "資料指紋":{"n_files":len(files),"combined_sha256":data},
        "程式指紋":{"pooled_ph.py":_sha(os.path.abspath(__file__)),
                    "cgm_lstm_markov.py":_sha(os.path.abspath(C.__file__))},
        "環境":{"python":sys.version.split()[0],"sys.platform":sys.platform,
                "machine":platform.machine(),"numpy":np.__version__,"pandas":pd.__version__,
                "tensorflow":tf.__version__,"keras":keras.__version__},
    }


FP=_fingerprint()


def _reset(reason):
    for p in (OUT,META):
        if os.path.exists(p): os.remove(p)
    print(f"  [重算] {reason}")


if _A.force:
    _reset(f"--force:已刪除既有之 {OUT},全部重算。")
elif os.path.exists(OUT):
    _prev=None
    if os.path.exists(META):
        try:
            _prev=json.load(open(META,encoding="utf-8")).get("fingerprint")
        except Exception:
            _prev=None
    if _prev is None:
        _reset(f"{OUT} 沒有對應之指紋檔 {META}(舊版程式之產物或已損毀),"
               f"無法確認其為同一設定、同一資料、同一程式之結果 → 刪除並全部重算。")
    elif _prev!=FP:
        _diff=[k for k in FP if _prev.get(k)!=FP.get(k)]
        _reset(f"{OUT} 之{('、'.join(_diff))}與本次不同 → 刪除並全部重算。")
done=set()
if os.path.exists(OUT):
    D0=pd.read_csv(OUT,float_precision="round_trip")   # 逐位元讀回,改寫時數值不變
    for k in range(n_folds):
        got=D0[D0.fold==k]
        if len(got)==0: continue
        if list(got.record.astype(str))==EXPECTED[k]:
            done.add(k)
        else:
            print(f"  [續跑] 第 {k+1} 折之紀錄不完整或不符(現有 {len(got)} 筆、應有 "
                  f"{len(EXPECTED[k])} 筆)→ 捨棄該折、重算。")
    keep=D0[D0.fold.isin(sorted(done))]
    if len(keep)!=len(D0):
        if len(keep): keep.to_csv(OUT,index=False)
        else: os.remove(OUT)
    if done:
        print(f"  [續跑] 第 {', '.join(str(k+1) for k in sorted(done))} 折已完成且通過檢查"
              f"(設定、資料、程式指紋相符;逐筆紀錄相符),不重算;其餘折重算。")
os.makedirs(os.path.dirname(META),exist_ok=True)
json.dump({"fingerprint":FP,
           "說明":"pooled_ph.py 續跑有效性檢查用:本檔之指紋與下次執行不同時,"
                  f"{os.path.basename(OUT)} 會被刪除並全部重算。請勿手動修改。"},
          open(META,"w",encoding="utf-8"),ensure_ascii=False,indent=1)

t0=time.time()
for k in range(n_folds):
    if k in done or time.time()-t0>BUDGET: continue
    C._keras_setup(C.SEED+k)   # 〔稽核修正 B-07〕每折獨立固定全部種子(續跑與一次跑完相同)
    te=EXPECTED[k]; tr=[r for r in ids if fold_of[P[r]["pid"]]!=k]
    tp=sorted({P[r]["pid"] for r in tr}); nv=max(1,len(tp)//10)
    vp=set(np.random.RandomState(C.SEED+100+k).permutation(tp)[:nv])
    fit=[r for r in tr if P[r]["pid"] not in vp]; val=[r for r in tr if P[r]["pid"] in vp]
    m=C.build_lstm_model(look_back,1,units)   # 〔稽核修正 B-12〕與其他 LSTM 共用架構定義
    m.compile(optimizer="adam",loss="mse")
    m.fit(np.concatenate([P[r]["Xtr"] for r in fit]),np.concatenate([P[r]["ytr"] for r in fit]),
          epochs=epochs,batch_size=bs,verbose=0,
          validation_data=(np.concatenate([P[r]["Xtr"] for r in val]),np.concatenate([P[r]["ytr"] for r in val])),
          callbacks=[EarlyStopping(patience=4,restore_best_weights=True)])
    rows=[]
    for r in te:
        d=P[r]; pred=m.predict(d["Xte"],verbose=0).ravel()*d["sd"]+d["mu"]
        act=d["yte"].ravel()*d["sd"]+d["mu"]; base=d["Xte"][:,-1,0]*d["sd"]+d["mu"]
        l_=float(np.sqrt(np.mean((pred-act)**2))); b_=float(np.sqrt(np.mean((base-act)**2)))
        rows.append(dict(record=r,patient=d["pid"],fold=k,PH_min=STEP*15,
            lstm_rmse=l_,lstm_mae=float(np.mean(np.abs(pred-act))),
            baseline_rmse=b_,baseline_mae=float(np.mean(np.abs(base-act))),
            skill_pct=(1-l_/b_)*100 if b_>0 else np.nan))
    pd.DataFrame(rows).to_csv(OUT,mode="a",header=not os.path.exists(OUT),index=False)
    print(f"  PH={STEP*15}m fold {k+1}/{n_folds} 完成 累計{time.time()-t0:.0f}s",flush=True)
# ---- [修正 25c] 折數完整性檢查 ----------------------------------------
if not os.path.exists(OUT):
    sys.exit(f"[★] PH={STEP*15}m 未產生任何輸出({OUT} 不存在)。")
D=pd.read_csv(OUT,float_precision="round_trip"); got=sorted(D.fold.unique())
# 〔稽核修正 B-09〕列排成固定順序(折號、紀錄),使續跑與一次跑完之檔案相同
_o={r:i for i,r in enumerate(ids)}
D2=D.assign(_o=D.record.astype(str).map(_o)).sort_values(["fold","_o"],kind="mergesort").drop(columns="_o")
if list(D2.index)!=list(D.index):
    D2.reset_index(drop=True).to_csv(OUT,index=False)
    print(f"  [整理] {OUT} 之列已依(折號、紀錄)排序。")
print(f"PH={STEP*15}m rows={len(D)} folds={got}")
if len(got)!=n_folds:
    print(f"[★] PH={STEP*15}m 只完成 {len(got)}/{n_folds} 折(時間預算 {BUDGET:g}s 用盡或中途失敗)。")
    print(f"     {OUT} 目前只含部分折之結果,下游彙總會把它當成完整結果使用,")
    print(f"     不得逕行引用。請重跑本檔(已完成之折會自動跳過)直到 {n_folds} 折齊全。")
    sys.exit(3)
# 〔稽核修正 B-09〕逐折檢查筆數與紀錄(原本只檢查折數)
_bad=[k for k in range(n_folds) if list(D2[D2.fold==k].record.astype(str))!=EXPECTED[k]]
if _bad or len(D2)!=len(ids):
    print(f"[★] PH={STEP*15}m 逐折檢查未通過:第 {', '.join(str(k+1) for k in _bad) or '—'} 折之紀錄不符,"
          f"總筆數 {len(D2)}(應為 {len(ids)})。請以 --force 重跑。")
    sys.exit(3)
print(f"  ✓ 逐折檢查通過:{n_folds} 折、每折紀錄與依病患分折之結果相符,共 {len(D2)} 筆。")
