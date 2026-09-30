# 糖尿病風險分層、CGM 動態分析與餘生醫療成本情境模擬

輔仁統計資訊學系 115 年第 26 屆學生專題成果發表會第六組

本專題使用 Shanghai_T2DM 公開資料，整合臨床併發症關聯分類與風險分層、同次監測 HbA1c 橫斷面估計、連續血糖監測（CGM）指標、血糖狀態 Markov 轉移、LSTM 短期血糖預測、長短期二維風險整合，以及 Monte Carlo 餘生醫療成本情境模擬。分析結果另以 GitHub Pages App 呈現，作為研究示範介面；模型尚未經臨床驗證，不能作為個別病患的診斷或治療依據。

分類結果是與目前已記錄併發症狀態相關的 risk stratification；HbA1c 分析使用同次臨床監測取得的其他特徵進行同次監測 HbA1c 橫斷面估計。成本結果是 exploratory model-based scenario simulation，不是實際觀察到的個別病患未來醫療帳單，也不是保險或正式醫療財務估價。

## 1. 資料來源與研究樣本

資料來自 [ShanghaiT2DM 公開資料集](https://doi.org/10.6084/m9.figshare.21600933)。請依資料頁的授權及說明自行取得資料。分析程式及公開 App 彙總資訊記載，臨床摘要包含 **109 筆紀錄、100 位病患**，CGM 分析包含 **109 份監測紀錄／檔案**，名目取樣間隔為 **15 分鐘**；部分病患有重複回診紀錄，因此紀錄數不等於病患數。

臨床資料以 `Patient Number` 第一個底線前的字串作為病患分組 ID。例如，同一前綴的多筆紀錄在分類模型交叉驗證中會分配到同一折，避免同一病患同時出現在訓練與測試資料。

CGM 指標包括平均血糖、標準差、CV、GMI、TIR、TBR 及 TAR。TIR 定義為 70–180 mg/dL（含端點），TBR 為低於 70 mg/dL，TAR 為高於 180 mg/dL。App 中的 cohort descriptive statistics 是先逐份監測計算指標，再彙總各監測紀錄的統計量；不是把所有 CGM 讀值合併後計算的單一病患值。

## 2. 臨床資料前處理

臨床建模使用 24 項特徵，涵蓋人口學與身體組成、吸菸／飲酒與糖尿病病程、血糖／胰島素／C-peptide／HbA1c／糖化白蛋白，以及血脂與腎功能指標。右偏變數使用 `log1p` 轉換後取代原值，不同時保留原值與轉換值；HbA1c、糖化白蛋白、HDL、LDL 等未列入右偏轉換的欄位保留原值。

缺失值以中位數插補，並放在 scikit-learn `Pipeline` 中；Logistic Regression 的 `StandardScaler` 也在 Pipeline 中。每次交叉驗證只用該訓練折 fit 前處理，避免測試折資訊洩漏。用藥相關二元特徵預設不納入（`INCLUDE_MED_FEATURES=False`），以降低因既有併發症或臨床處置導致的 reverse causality／答案洩漏風險。

## 3. 臨床併發症關聯分類

主要分類目標為 `Any_Complication`、`Macrovascular` 及 `Microvascular`。Hypoglycemia 因陽性樣本較少，不列為主要模型比較目標。分類模型以 Logistic Regression 作為較易解釋的基準模型，並以 XGBoost 作為非線性 challenger；若 XGBoost 不可用，程式退回 HistGradientBoosting。可用勝算比（odds ratio）及 SHAP 或 permutation importance 輔助解釋模型關聯，結果用於 complication-associated risk score 與 risk stratification。

保留全部 109 筆紀錄，以 **StratifiedGroupKFold 五折、100 次重複**進行病患分組交叉驗證；各次使用固定種子序列 `SEED + r`。以下為公開 App 顯示的 Logistic Regression AUC 中位數及 100 次重複結果的 empirical 2.5–97.5 百分位區間。此區間描述重複分折的變異，不是外部驗證信賴區間。

| Outcome | AUC 中位數 | 100 次重複之 empirical 2.5–97.5 百分位 |
|---|---:|---:|
| 任一併發症 | 0.862 | 0.800–0.903 |
| 小血管併發症 | 0.829 | 0.771–0.870 |
| 大血管併發症 | 0.839 | 0.790–0.880 |

## 4. 同次監測 HbA1c 橫斷面估計

本分析使用同次臨床監測所取得的其他臨床特徵，進行同次監測 HbA1c 橫斷面估計（cross-sectional estimation）。

目標為 HbA1c，特徵中排除 HbA1c 本身及糖化白蛋白（GA），避免直接納入目標或高度近似的血糖指標。比較 Linear Regression baseline 與 XGBoost Regression；未安裝 XGBoost 時使用 HistGradientBoosting。以病患為單位分配至五折並重複 100 次，評估 MAE、RMSE、MAPE 及 R²。所有插補及模型前處理均在各訓練折內完成。

目前公開 checkout 沒有可直接核對的最終 `reg_results.csv`，App 也沒有呈現可核對的 HbA1c 迴歸效能數字，因此本 README 僅記錄方法，不列迴歸指標。

## 5. CGM 指標與 Markov 血糖狀態分析

以 Low（<70 mg/dL）、InRange（70–180 mg/dL）及 High（>180 mg/dL）三種狀態建立一階 Markov 轉移矩陣。僅計入時間戳相鄰且間隔不超過 30 分鐘的讀值轉移；較長的間隔視為監測中斷，不跨越斷點建立轉移。下表為公開 App 可核對的 cohort-level 描述值與轉移摘要：

| 指標 | Cohort 平均 |
|---|---:|
| 平均血糖 | 141.66 mg/dL |
| CV | 28.36% |
| GMI | 6.70% |
| TIR | 77.68% |
| TBR | 2.36% |
| TAR | 19.96% |

CGM 指標是研究樣本監測紀錄的描述統計，不是個別病患的臨床建議。由整體轉移矩陣可見，Low→Low 約 82.9%、InRange→InRange 約 97.3%、High→High 約 90.8%；該矩陣的 stationary distribution 約為 Low 2.3%、InRange 78.9%、High 18.8%。這些是研究樣本的 Markov 描述與模型推導值，不代表臨床結果保證。

### LSTM 短期血糖預測

LSTM 使用長度 12 的回看視窗（名目取樣間隔下約為過去 3 小時），架構為 64 個 LSTM units、Dropout(0.2) 及 Dense 輸出層，並與 persistence baseline（下一筆血糖等於目前血糖）比較。逐病患模型依每份監測序列的時間順序，以前 80% 訓練、後 20% 測試；跨病患 pooled 模型則以病患分組五折評估，測試病患不參與該折模型訓練。測試視窗仍使用該病患自身先前序列作為輸入歷史，且以該病患訓練段統計量標準化。

公開 App 可核對的逐筆比較摘要如下；改善率為相對 persistence baseline 的 RMSE 改善：

| 預測時界 | 可核對結果 |
|---|---|
| 15 分鐘，pooled LSTM | RMSE 改善中位數約 37.1%；約 99.1% 監測紀錄的 RMSE 優於 persistence baseline |
| 15 分鐘，逐病患模型 | RMSE 改善中位數約 20.9% |

30 分鐘及其他預測時界以相同架構比較 LSTM 與 persistence baseline；目前 README 不列這些時界的效能數字。

## 6. 長短期二維風險分層

長期軸是 Logistic Regression 對 `Any_Complication` 的病患分組交叉驗證 out-of-fold 機率，五折、重複 100 次後以各次 out-of-fold 機率平均形成 complication-associated risk score，用於既有併發症狀態的關聯分層。

短期軸 `short_risk` 由每位病患自己的 Markov transition matrix 推算，固定以 InRange 為起點，計算指定時界後落入 Low 或 High 的機率總和。**主要短期風險軸為 `SHORT_HORIZON = 30m`**：亦即 30 分鐘後落入 Low（<70）或 High（>180）的機率總和。另保留 15、60、90 分鐘及 4、24 小時時界，用於敏感度與 Markov 收斂性比較；四象限主要短期軸採 30 分鐘。

固定 InRange 起點的 `short_risk` 用於研究分層；`*_fromlast` 則由最後一筆實際血糖狀態起算，用於即時情境／App 走勢呈現。兩者起點與用途不同，不應混為同一風險量。

兩軸各以中位數切分高低，四象限的公開 App 摘要如下。N 為臨床紀錄數；比例表示該象限中目前已有／觀察到併發症的紀錄比例，屬橫斷面摘要。

| 象限 | 長期／短期分層 | N（紀錄） | 該象限中目前已有／觀察到併發症的紀錄比例 |
|---|---|---:|---:|
| A | 高／高 | 34 | 73.5% |
| B | 高／低 | 21 | 61.9% |
| C | 低／高 | 21 | 9.5% |
| D | 低／低 | 33 | 9.1% |

### App 部署分數與研究分層的差別

交叉驗證用於研究效能估計與分層門檻建立；研究 AUC 與 long-risk 分層門檻／評估均來自病患分組的 out-of-fold evaluation。App 部署模型則以完整研究資料（109 筆紀錄）重新配適 Logistic Regression 係數，用於新輸入資料之展示性風險計算。App 單一使用者的輸出不是 out-of-fold prediction，也不代表該個人分數已經過外部驗證。App 建置核對另記錄兩種分組在 109 筆紀錄中有 8 筆（7.3%）落入不同象限，故 App 個別部署畫面不等同於研究交叉驗證分層結果。

## 7. 餘生醫療成本情境模擬

年直接醫療成本基礎取自陳興寶等（2003）的糖尿病成本資料，分為 none、macro、micro、both 四種併發症狀態。原始成本為 2001 年幣值，以固定 CPI multiplier **1.3737** 調整至 2024 年。Monte Carlo 的不確定性主要來自 complication state 抽樣及成本分布等模型成分；目前 final implementation 中 CPI adjustment 為 deterministic（`INFLATION_CV=0.0`），成本參數以 Gamma 分布抽樣（`COST_CV=0.30`）。

存活與折現使用中國精算師協會發布的 2025 中國人身保險業經驗生命表非養老類業務表，男女分表，discount rate 為 **3%**。不把平均餘命直接當作確定存活年數，而採 survival-weighted annuity：

`a_x = Σ (v^t × tpx),  t = 1, 2, ...`

其中 `v = 1 / (1 + r)` 為折現因子，`tpx` 為病患存活至第 `t` 年的機率。每位病患執行 **5,000 次**模擬，依模型估計的 macro／micro probabilities 分別抽樣併發症狀態，再套用各狀態成本與存活加權年金。population-level total 僅計每位病患一筆紀錄，重複回診不重複計入；程式依來源表格順序選取每位病患的第一筆紀錄。CPI 在目前實作中是確定性校正，Monte Carlo 不抽樣通膨倍數。

公開 App 中可核對的象限餘生成本平均與 95% 模擬區間如下。這是 **exploratory model-based scenario simulation**，不是實際觀察到的個別病患未來醫療帳單，也不是保險或正式醫療財務估價。

| 象限 | 平均餘生成本 | 95% 模擬區間 |
|---|---:|---:|
| A | 約 42.2 萬 RMB | 25.4–66.6 萬 RMB |
| B | 約 45.0 萬 RMB | 26.6–71.9 萬 RMB |
| C | 約 14.1 萬 RMB | 7.8–22.6 萬 RMB |
| D | 約 14.8 萬 RMB | 8.7–22.4 萬 RMB |

## 8. GitHub Pages 決策支援 App

[開啟糖三臟研究示範 App](https://itzu-huang.github.io/Diabetes_Deterioration_Risk_Project/)

<p align="center">
  <a href="https://itzu-huang.github.io/Diabetes_Deterioration_Risk_Project/">
    <img src="qrcode_app.png" alt="糖三臟決策支援 App QR Code" width="220">
  </a>
</p>

<p align="center">
  <strong>掃描 QR Code 或點擊圖片即可開啟 App</strong>
</p>

GitHub Pages 使用 `main` 分支的 `/docs`。`docs/index.html` 以相對路徑導向 App；Excel 解析使用同目錄內的 `vendor/xlsx.full.min.js`。使用者匯入的資料在瀏覽器內處理，不上傳至外部伺服器。App 整合臨床併發症關聯風險、CGM 短期風險、四象限分層與餘生成本情境展示；未經臨床驗證，不可作為診斷或治療依據。

## 9. Repository 結構

```text
code/           分析、整合流程與輸出驗證程式
verification/   四層驗證入口、交付物核對腳本、豁免規則
docs/           GitHub Pages 首頁與 App
data/           本機資料目錄（原始資料不納入版本控制）
output/         公開結帳僅保留目錄結構；分析輸出留在本機
requirements.txt
.gitignore
```

主要分析程式：

- `diabetes_deterioration_pipeline.py`：臨床資料前處理、分類與風險關聯分析
- `diabetes_regression.py`：HbA1c 橫斷面估計
- `cgm_lstm_markov.py`：CGM 指標、Markov 與逐病患 LSTM
- `pooled_ph.py`：pooled LSTM 預測時界分析
- `deterioration_risk.py`：長短期二維風險與四象限分層
- `lifetime_cost_montecarlo.py`：餘生醫療成本情境模擬
- `run_all.py`：依相依順序執行分析與記錄重現資訊

報告、簡報與海報公開副本仍需完成去識別及版面驗收後才會加入。原始資料、病患層級輸出、凍結快照、執行日誌和本機環境資訊不隨公開 repo 發布；處理後的病患層級檔案也不可提交。

## 10. 安裝與重現分析

來源交付環境使用 Python 3.13.9。`requirements.txt` 保留該環境的主要分析套件版本，另包含文件驗證所需的 `python-docx`、`python-pptx` 與 `Pillow`。

```sh
python -m venv .venv
# Windows PowerShell
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

資料來源及取得方式見上方 figshare 資料頁。將 `Shanghai_T2DM_Summary.xlsx` 與 `Shanghai_T2DM/` CGM 資料夾放在 `code/` 後執行：

```sh
cd code
python run_all.py --with-sensitivity
```

`run_all.py` 依序執行臨床分類、HbA1c 迴歸、CGM／Markov／逐病患 LSTM、pooled 預測時界、時界彙總、二維風險、成本模擬及圖表；指定 `--with-sensitivity` 時，再執行缺失值敏感度、EPV、診斷、共線性及其他敏感度分析。完整流程包含深度學習，耗時依電腦而異。`--clean` 會清除程式輸出，僅在確實要從頭重跑且已保留必要結果時使用。

完整執行會在 `code/` 產生 `data_manifest.json`、`env_lock.json` 及 `run_log.json`，記錄資料 SHA-256、Python／套件環境、程式 SHA-256、實際 argv、輸出檔及雜湊。這些本機執行產物不應提交。固定亂數種子有助重現，但不保證跨作業系統 bit-for-bit 相同；XGBoost、TensorFlow／LSTM 及硬體浮點運算仍可能造成差異。使用 `--skip-lstm` 會略過 LSTM 訓練，不應將該次執行描述為完整重現。

## 11. 驗證機制

```sh
python verification/verify_all.py --stage-dir code --deliverables-dir docs --scripts-dir verification
```

完整核對需要已驗收的報告／簡報／海報、分析輸出與獨立完整執行的凍結快照。公開 GitHub checkout 不含這些資料時，缺少資料應回報未執行或失敗，不可宣稱公開 repo 已通過完整四層驗證。

1. 比較本次輸出與另一完整執行的凍結快照。
2. 核對交付文件及 App 中的研究數字是否對應分析輸出。
3. 反向掃描交付物中的研究數字是否能在來源輸出中找到；找到來源不等於每個引用位置都正確。
4. 核對已涵蓋表格格位的四捨五入結果。

來源交付物曾通過四層檢查；公開副本的驗證狀態需另行記錄，不能沿用原件結果。

## 12. 研究限制

- 樣本約 100 位病患，來自單一 Shanghai_T2DM cohort；需以獨立外部資料驗證泛化能力。
- 臨床併發症目標是橫斷面既有狀態；分類結果用於關聯分析、complication-associated risk score 與 risk stratification。
- HbA1c 分析範圍限於同次監測 HbA1c 橫斷面估計。
- Markov 模型採一階轉移假設，並排除相鄰 CGM timestamp gap >30 minutes 的轉移。LSTM window construction 主要依讀值順序建立 windows，沒有相同的 timestamp-gap filtering；長時間資料中斷可能限制 LSTM sequence interpretation。
- LSTM 結果受序列長度、分折、裝置及計算環境影響；測試視窗仍使用該監測序列較早的讀值作為輸入歷史。
- 用藥特徵預設排除，以降低 reverse causality／答案洩漏風險；本研究無法由橫斷面關聯確立因果方向。
- 成本情境依賴歷史成本、上海 CPI 校正、生命表、模型機率、成本變異與折現假設；不是實際未來支出的觀察值或正式財務估價。
- App 是研究成果展示介面，尚未經臨床驗證，不能取代醫療專業判斷。

## 公開範圍與隱私

`.gitignore` 排除資料、逐筆輸出、模型檔、快照、壓縮包、環境秘密檔與執行日誌。忽略規則不會移除已追蹤檔案，因此提交前仍須檢查 staged diff。不要使用強制加入來繞過資料排除規則。
