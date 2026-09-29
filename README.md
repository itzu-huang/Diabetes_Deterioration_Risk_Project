# 糖尿病惡化風險分層、CGM 短期預測與醫療成本情境分析

本專案整合第二型糖尿病臨床摘要資料與連續血糖監測（Continuous Glucose Monitoring, CGM）資料，建立包含下列任務的研究流程：

- 臨床併發症關聯分類與風險分層；
- HbA1c 橫斷面迴歸；
- K-means 與 Gaussian Mixture Model（GMM）探索性分群；
- CGM 指標、AGP 與 Markov 狀態轉移；
- 單一監測、逐監測與跨病患 pooled LSTM；
- 臨床與 CGM 二維風險整合；
- 探索性餘生前瞻醫療成本情境模擬。

本專案為 undergraduate research / coursework prototype，尚未經臨床驗證，不可用於診斷、治療或個別病患之未來臨床事件預測，也不應作為正式醫療資源配置依據。

---

## 線上決策支援 App

本專案另提供「糖三臟」高齡友善糖尿病決策支援 App 原型，可直接透過 GitHub Pages 開啟，無須下載或安裝程式。

### 開啟 App

👉 [點此開啟糖三臟決策支援 App](https://itzuakatsuki.github.io/Diabetes_Deterioration_Risk_Project/)

亦可使用手機掃描下方 QR Code：

<p align="center">
  <a href="https://itzuakatsuki.github.io/Diabetes_Deterioration_Risk_Project/">
    <img src="qrcode_app.png" alt="糖三臟決策支援 App QR Code" width="220">
  </a>
</p>

<p align="center">
  <strong>掃描 QR Code 或點擊圖片即可開啟 App</strong>
</p>

---

## 1. 研究定位

### 1.1 臨床資料

臨床摘要資料屬於橫斷面資料。臨床特徵與併發症狀態是在相近時間點記錄，因此分類模型估計的是：

> 目前臨床特徵與資料中已有大血管或小血管併發症者之間的關聯與相似程度。

臨床模型適合解讀為：

- 目前併發症的關聯分類；
- 病患風險分層；
- 追蹤與進一步檢查的優先排序參考。

臨床模型不應解讀為：

- 未來 1、3 或 5 年併發症發生率；
- 中風、腎病、視網膜病變或神經病變的確定機率；
- 因果效果；
- 臨床診斷。

### 1.2 CGM 資料

CGM 序列具有真實時間順序，因此 Markov 狀態轉移與 LSTM 下一步血糖預測才是本專案中真正具有時間性的分析。

但是，時間性分析仍受到下列限制：

- 部分函式未嚴格檢查每筆資料是否正好間隔 15 分鐘；
- LSTM 視窗可能跨越缺測區段；
- Markov 齊次性與狀態轉移穩定性尚未正式驗證。

### 1.3 成本分析

成本程式將橫斷面併發症風險分數、文獻年度成本、折現與 Monte Carlo 模擬結合，形成探索性的餘生前瞻醫療成本情境。

此結果不是：

- 已驗證的個人餘生成本預測；
- 完整疾病微觀模擬；
- 含逐年併發症進展與死亡競爭風險的健康經濟模型。

---

## 2. 資料結構

目前研究資料包含：

- 109 筆臨床監測紀錄；
- 約 100 位獨立第二型糖尿病病患；
- 部分病患具有多次回診紀錄；
- 109 份 CGM 監測檔；
- CGM 原則上約每 15 分鐘記錄一次血糖。

資料來源：ShanghaiT2DM dataset，Figshare（CC BY 4.0）。Zhao Q, Zhu J, Shen X, et al. Chinese diabetes datasets for data-driven machine learning. *Scientific Data* 10, 35 (2023). [Article](https://doi.org/10.1038/s41597-023-01940-7) · [Dataset](https://doi.org/10.6084/m9.figshare.c.6310860).

主要分類目標：

| Target | 說明 | 本研究定位 |
|---|---|---|
| `Macrovascular` | 是否已有大血管併發症 | 主要分類目標 |
| `Microvascular` | 是否已有小血管併發症 | 主要分類目標 |
| `Any_Complication` | 是否已有任一類併發症 | 綜合分層代理指標 |
| `Hypoglycemia` | 是否有低血糖紀錄 | 陽性數較少，僅供補充 |

原始病患資料、CGM 檔案與含病患識別碼的輸出不應提交至公開 repository。

---

## 3. X 與 Y

### 3.1 臨床分類

#### X：臨床特徵

模型使用的人口學、身體組成、生活型態、病程與生化檢驗特徵包括：

- `Age`
- `Height`
- `Weight`
- `BMI`
- `Gender`
- `Smoking`
- `Alcohol`
- `Duration`
- `FPG`
- `PPG`
- `Cpep_f`
- `Cpep_2h`
- `Ins_f`
- `Ins_2h`
- `HbA1c`
- `GA`
- `TC`
- `TG`
- `HDL`
- `LDL`
- `Cr`
- `eGFR`
- `UA`
- `BUN`

`Patient Number` 不作為模型特徵，只用來建立病患分組。

用藥與共病欄位預設不納入主要 X，因為部分用藥可能是在併發症已出現後才開立，容易形成反向因果或答案洩漏。

#### Y：二元標籤

| Target | Y = 0 | Y = 1 |
|---|---|---|
| `Macrovascular` | 無大血管併發症 | 有大血管併發症 |
| `Microvascular` | 無小血管併發症 | 有小血管併發症 |
| `Any_Complication` | 無納入定義的併發症 | 至少一種併發症 |

程式定義：

```text
Any_Complication
= Macrovascular OR Microvascular OR Acute Diabetic Complications
```

目前資料中的 `Acute Diabetic Complications` 全為 0，因此實際等同：

```text
Any_Complication
= Macrovascular OR Microvascular
```

這些標籤代表該次紀錄中目前是否已有併發症，不代表未來新發事件。

### 3.2 HbA1c 迴歸

預設設定：

- Y：`HbA1c`
- X：排除 `HbA1c` 與 `GA` 後的其餘臨床特徵
- 基準模型：Linear Regression
- 主要模型：XGBoost Regressor
- XGBoost 不可用時：HistGradientBoostingRegressor

此任務估計的是同次紀錄中的 HbA1c，不是未來 HbA1c 的縱向預測。

若更換 `TARGET`，必須同步：

1. 將新目標從特徵中排除；
2. 檢查 `load_data()` 是否對該欄位進行 `log1p`；
3. 重新確認近似目標或高度相關欄位是否需要排除。

目前 TIR／GMI 與臨床資料的合併流程尚未自動實作。

### 3.3 LSTM

預設：

```text
X = 過去 12 筆 CGM
Y = 下一筆 CGM
```

若每筆資料確實相隔 15 分鐘：

- 過去 12 筆約為過去 3 小時；
- 下一筆約為 15 分鐘後血糖。

### 3.4 分群

K-means 與 GMM 為非監督式分析：

- 有 X；
- 沒有 Y；
- 併發症標籤只在分群完成後，用於探索群組間比例差異。

---

## 4. 資料前處理

### 4.1 缺失值

原始資料可能以 `/` 表示缺失。程式先轉為缺失值，再於模型 Pipeline 中使用：

```python
SimpleImputer(strategy="median")
```

插補器只在每一折的訓練資料上配適，避免測試折資訊提前進入模型。

### 4.2 EDA 與建模尺度分離

`load_data()` 在 `log1p` 前保存：

```python
d_raw = d.copy()
```

因此：

- `d_raw`：保留原始臨床單位，供 EDA 與描述性統計使用；
- `d`：進行必要轉換後，供分類、迴歸與風險模型使用。

### 4.3 對數轉換

下列右偏變數以 `log1p` 取代原值：

- `FPG`
- `PPG`
- `Cpep_f`
- `Cpep_2h`
- `Ins_f`
- `Ins_2h`
- `TG`
- `TC`
- `Cr`
- `eGFR`
- `UA`
- `BUN`
- `Duration`

`HbA1c`、`GA`、`HDL` 與 `LDL` 保留原尺度。

### 4.4 模型 Pipeline

| 模型 | 前處理 |
|---|---|
| Logistic Regression | Median imputation + StandardScaler |
| Linear Regression | Median imputation + StandardScaler |
| XGBoost／HistGradientBoosting 分類 | Median imputation |
| XGBoost／HistGradientBoosting 迴歸 | Median imputation |

### 4.5 病患層級洩漏

預設：

```python
DEDUP_ONE_PER_PATIENT = False
USE_GROUP_CV = True
```

代表：

- 保留全部 109 筆紀錄；
- 以病患主編號建立 GroupKFold；
- 同一病患的多次紀錄不會同時出現在訓練折與驗證折。

若將 `DEDUP_ONE_PER_PATIENT` 改為 `True`，程式會每位病患只保留第一筆紀錄，並改用 RepeatedStratifiedKFold。

### 4.6 OOF 預測

分類、迴歸與風險引擎使用 out-of-fold（OOF）預測。每筆 OOF 預測來自未使用該驗證資料訓練的模型。

分類表中的 AUC、AP、Accuracy、Precision、Recall 與 F1 為交叉驗證各折指標的平均與標準差；ROC 圖則使用合併後的 OOF 預測計算 pooled OOF AUC，兩者數值不必完全相同。

---

## 5. Label 與狀態定義

### 5.1 臨床類別編碼

| Variable | Raw value | Model value |
|---|---|---:|
| `Gender` | Female = 1 | 0 |
| `Gender` | Male = 2 | 1 |
| `Smoking` | pack-year = 0 或缺失 | 0 |
| `Smoking` | pack-year > 0 | 1 |
| `Alcohol` | non-drinker | 0 |
| `Alcohol` | drinker | 1 |
| `Hypoglycemia` | no | 0 |
| `Hypoglycemia` | yes | 1 |
| Complication／medication fields | none／no／nan／空字串 | 0 |
| Complication／medication fields | 有其他紀錄 | 1 |

### 5.2 CGM 三狀態

| Code | Label | 血糖範圍 |
|---:|---|---|
| 0 | Low | `< 70 mg/dL` |
| 1 | In Range | `70 <= glucose <= 180 mg/dL` |
| 2 | High | `> 180 mg/dL` |

### 5.3 CGM 五狀態

| Code | Label | 血糖範圍 |
|---:|---|---|
| 0 | Very Low | `< 54 mg/dL` |
| 1 | Low | `54 <= glucose < 70 mg/dL` |
| 2 | In Range | `70 <= glucose <= 180 mg/dL` |
| 3 | High | `180 < glucose <= 250 mg/dL` |
| 4 | Very High | `> 250 mg/dL` |

---

## 6. 主要程式

### 6.1 `code/diabetes_deterioration_pipeline.py`

功能：

- 讀取臨床 Excel；
- 擷取病患主編號；
- 類別欄位編碼；
- 建立 `d_raw` 與建模資料；
- 右偏變數 `log1p`；
- 建立分類目標；
- EDA；
- Logistic Regression；
- XGBoost／HistGradientBoosting；
- GroupKFold OOF 評估；
- ROC 與混淆矩陣；
- SHAP 或 permutation importance；
- Logistic Regression 標準化係數與 Odds Ratio；
- K-means；
- GMM；
- silhouette、BIC、PCA 與探索性卡方檢定。

設定：

```python
SEED = 42
XLSX_PATH = "Shanghai_T2DM_Summary.xlsx"
OUTDIR = "output"
DEDUP_ONE_PER_PATIENT = False
USE_GROUP_CV = True
INCLUDE_MED_FEATURES = False
RUN_EDA = True
```

注意：

- `IMP_{target}.png` 會固定產生；
- 若 XGBoost 與 SHAP 可用，另產生 `SHAP_{target}.png`；
- 特徵重要性與 Odds Ratio 是在全資料重新配適後產生，屬探索性解釋，不是外部驗證或因果效果；
- Odds Ratio 為正則化 Logistic Regression 的標準化係數指數值，不能等同傳統未懲罰模型的推論 OR。

### 6.2 `code/diabetes_regression.py`

功能：

- Linear Regression 基準模型；
- XGBoost Regressor 主要模型；
- HistGradientBoostingRegressor 備援模型；
- GroupKFold OOF 預測；
- MAE、RMSE、MAPE、R²；
- actual vs predicted；
- residual plot；
- permutation importance。

設定：

```python
SEED = 42
OUTDIR = "output_reg"
TARGET = "HbA1c"
EXCLUDE_AS_FEATURE = ["HbA1c", "GA"]
```

Permutation importance 目前是在全資料配適後計算，屬探索性解釋。

### 6.3 `code/cgm_lstm_markov.py`

功能：

- 讀取 `.xls` 與 `.xlsx`；
- 計算 Mean、SD、CV、GMI、TIR、TBR、TAR；
- 3-state 與 5-state pooled Markov matrix；
- stationary distribution；
- 個別 CGM 波動圖；
- pooled AGP；
- 12 筆監測示範網格；
- CGM 指標分布與描述性統計；
- 單一監測 LSTM；
- 全體監測 persistence baseline；
- 每份監測分別訓練 LSTM；
- 跨病患 pooled LSTM；
- 依病患分組的 pooled LSTM 交叉驗證；
- pooled、per-record LSTM 與 persistence baseline 比較。

設定：

```python
SEED = 42
CGM_DIR = "Shanghai_T2DM"
OUTDIR = "cgm_output"
```

#### 母體 Markov

預設：

```python
markov_matrix(n_states=3, max_gap_min=30)
```

只要兩筆時間差不超過 30 分鐘就可能被計為一次轉移，因此不是嚴格的 15 分鐘轉移。

正式分析可考慮只保留：

```text
14 <= gap_minutes <= 16
```

#### 單一監測與逐監測 LSTM

- 前 80% 為訓練段；
- 後 20% 為測試段；
- 標準化只使用訓練段統計量；
- 目前 `validation_split=0.15` 未明確設定 `shuffle=False`；
- `make_windows()` 未檢查 timestamp，視窗可能跨缺測區段。

#### Pooled LSTM

pooled 版本：

- 將多位訓練病患的視窗合併；
- 依病患主編號分為 5 折；
- 同一病患的多次監測置於同一折；
- 訓練折內另保留病患層級 validation 供 early stopping；
- 模型權重未使用測試折病患資料；
- 但每位測試監測仍使用自身前 80% CGM 估計標準化平均與標準差，因此不是完全零歷史資料的 cold-start 預測。

直接執行主程式會同時嘗試執行單一監測、逐監測及 pooled LSTM，運算時間可能較長。

### 6.4 `code/deterioration_risk.py`

功能：

- 長期軸：Logistic Regression 對 `Any_Complication` 的 GroupKFold OOF score；
- 短期軸：每份 CGM 的 3-state Markov k-step scenario score；
- 低／中／高三分位分組；
- 以兩軸中位數建立四象限；
- 輸出風險表、二維圖與象限摘要。

#### 長期軸

長期軸使用：

```python
LogisticRegression(class_weight="balanced")
```

其 OOF `predict_proba()` 適合作為相對風險排序分數，不應直接視為已校準的臨床絕對機率。

#### 短期軸

程式將：

- 4 步標記為 1 小時；
- 16 步標記為 4 小時；
- 96 步標記為 24 小時。

但目前 `transition_matrix()` 沒有實際使用 timestamp，因此這些值應解讀為：

> 假設每一步均為 15 分鐘時的 k-step scenario score。

此外：

- 預測起點固定為 `InRange`；
- 目前不使用病患最後一筆真實狀態；
- 第 96 步不保證已達穩態；
- 四象限依目前樣本中位數切分，不是外部驗證的臨床門檻。

### 6.5 `code/lifetime_cost_montecarlo.py`

功能：

- Macrovascular 與 Microvascular OOF scores；
- Bernoulli 併發症狀態抽樣；
- Gamma 年度成本參數不確定性；
- 上海醫療保健類 CPI 逐年連乘；
- 折現年金；
- 生命表餘命或固定年齡情境；
- Monte Carlo PSA；
- 每筆紀錄成本平均與 95% 模擬區間；
- 去重後母體總成本分布；
- Lorenz curve 與 Gini；
- balanced／unweighted class-weight 敏感度分析；
- AUC 與 Brier score 比較。

設定：

```python
SEED = 42
N_SIM = 5000
COST_TYPE = "direct"
COST_CV = 0.30
COST_BASE_YEAR = 2001
TARGET_YEAR = 2024
INFLATION_MEAN = cumulative_inflation()
INFLATION_CV = 0.0
DISCOUNT_RATE = 0.03
HORIZON_MODE = "life_table"
HORIZON_AGE = 80
CLASS_WEIGHT = "balanced"
CLASS_WEIGHT_SCENARIOS = ["balanced", None]
OUTDIR = "output_cost_mc"
```

#### 成本與幣值

文獻發表於 2003 年，但表中成本幣值為 2001 年人民幣。

程式內列示 2002–2024 年上海醫療保健相關 CPI，並以逐年連乘方式將 2001 年成本校正至 2024 年幣值。

```python
INFLATION_CV = 0.0
```

代表 CPI 目前視為固定情境參數，不再額外進行 Gamma 抽樣。

正式研究文件仍應列出：

- 各年度官方來源；
- 表號；
- 下載日期；
- 完整連乘計算表；
- 2015／2016 年分類口徑變更。

#### 成本時界

預設：

```python
HORIZON_MODE = "life_table"
```

但 `LIFE_EXPECTANCY` 目前仍是暫定年齡別餘命情境，不是已核定的正式官方生命表。

若切換為：

```python
HORIZON_MODE = "fixed_age"
```

目前程式對年齡已超過 `HORIZON_AGE` 者仍強迫保留 1 年，這是已知限制。

#### 重複紀錄

- 個人 CSV 保留全部臨床紀錄；
- 母體總成本只納入每位病患首筆紀錄；
- 每位病患排序圖與 Lorenz curve 使用去重後資料；
- 目前主畫面的「每人平均」、class-weight sensitivity 中的每人平均，以及象限成本彙整仍可能使用全部紀錄，因此可能讓重複回診病患取得較高權重。

#### 模型假設

- Macro 與 Micro 分開抽樣，隱含給定 X 後條件獨立；
- 橫斷面風險分數被外推至整個成本時界；
- 沒有逐年疾病進展；
- 沒有死亡競爭風險；
- 成本結果只能視為探索性情境分析。

---

## 7. 模型與評估指標

| Task | Baseline | Main model／method | Metrics |
|---|---|---|---|
| Classification | Logistic Regression | XGBoost／HistGradientBoosting | AUC、AP、Accuracy、Precision、Recall、F1 |
| Regression | Linear Regression | XGBoost Regressor／HistGradientBoostingRegressor | MAE、RMSE、MAPE、R² |
| Clustering | K-means | GMM | Silhouette、BIC、探索性卡方檢定 |
| Time series | Persistence baseline | Per-record LSTM、pooled LSTM、Markov | RMSE、MAE、Skill |
| Cost | 文獻成本情境 | Monte Carlo PSA | Mean、2.5% 與 97.5% 模擬百分位、Brier、AUC |

README 不列入未經目前 repository 輸出重新驗證的模型數字。

---

## 8. Repository Structure

```text
Diabetes_Deterioration_Risk_Project/
├── code/
│   ├── diabetes_deterioration_pipeline.py
│   ├── diabetes_regression.py
│   ├── cgm_lstm_markov.py
│   ├── deterioration_risk.py
│   └── lifetime_cost_montecarlo.py
├── data/
│   ├── clinical/
│   └── cgm/
├── output/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

目前五支程式仍使用 legacy relative paths，未自動使用 `data/` 與統一的 `output/` 子目錄。

---

## 9. 安裝方式

建議使用 Python 3.10 以上版本。

建立虛擬環境：

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

macOS／Linux：

```bash
source .venv/bin/activate
```

更新 pip：

```bash
python -m pip install --upgrade pip
```

安裝套件：

```bash
python -m pip install -r requirements.txt
```

---

## 10. 資料位置

現行程式預期：

```text
Shanghai_T2DM_Summary.xlsx
Shanghai_T2DM/
```

輸出：

```text
output/
output_reg/
cgm_output/
output_risk/
output_cost_mc/
```

若資料放在 `data/clinical/` 與 `data/cgm/`，需修改程式常數或建立統一設定檔。

---

## 11. 執行順序

在 repository 根目錄執行：

```bash
python code/diabetes_deterioration_pipeline.py
python code/diabetes_regression.py
python code/cgm_lstm_markov.py
python code/deterioration_risk.py
python code/lifetime_cost_montecarlo.py
```

依賴關係：

- regression、risk 與 cost 會 import `load_data()`；
- 三者依賴 `diabetes_deterioration_pipeline.py`；
- risk 需要臨床 Excel 與 CGM；
- cost 若找到 `output_risk/patient_risk_table.csv`，會合併象限欄位；
- 若 risk 程式尚有 `long_risk` 欄位命名不一致，應先修正後再執行 risk 與 cost。

CGM 主程式會執行多個 LSTM，運算時間可能較長。

---

## 12. 輸出檔案

### 12.1 臨床分類、EDA 與分群

```text
output/
├── model_results.csv
├── ROC_Any_Complication.png
├── ROC_Microvascular.png
├── ROC_Macrovascular.png
├── CM_Any_Complication.png
├── CM_Microvascular.png
├── CM_Macrovascular.png
├── IMP_Any_Complication.png
├── IMP_Microvascular.png
├── IMP_Macrovascular.png
├── importance_Any_Complication.csv
├── importance_Microvascular.csv
├── importance_Macrovascular.csv
├── lr_oddsratio_Any_Complication.csv
├── lr_oddsratio_Microvascular.csv
├── lr_oddsratio_Macrovascular.csv
├── Clustering_PCA.png
└── eda_figures/
    ├── 00_target_distribution.png
    ├── 01_missing_heatmap.png
    ├── 02_histograms.png
    ├── 03_boxplots.png
    ├── 04_correlation_matrix.png
    ├── missing_pct.csv
    ├── correlation_matrix.csv
    └── describe.csv
```

若 XGBoost 與 SHAP 可用，另外輸出：

```text
SHAP_Any_Complication.png
SHAP_Microvascular.png
SHAP_Macrovascular.png
```

### 12.2 HbA1c 迴歸

```text
output_reg/
├── reg_results.csv
├── reg_pred_vs_actual_HbA1c.png
├── reg_residuals_HbA1c.png
├── reg_importance_HbA1c.csv
└── reg_importance_HbA1c.png
```

### 12.3 CGM、Markov 與 LSTM

```text
cgm_output/
├── cgm_metrics.csv
├── cgm_descriptive.csv
├── markov_3state.csv
├── markov_3state.png
├── markov_5state.csv
├── markov_5state.png
├── AGP_all.png
├── fluctuation_grid12.png
├── cgm_metrics_dist.png
├── fluctuation_<record>.png
├── baseline_all.csv
├── lstm_forecast_<patient>.png
├── lstm_vs_baseline_all.csv
├── lstm_vs_baseline_all.png
├── lstm_pooled_vs_baseline.csv
└── lstm_pooled_vs_perpatient.png
```

LSTM 檔案只在 TensorFlow 可用且程序成功執行時產生。

### 12.4 風險引擎

```text
output_risk/
├── patient_risk_table.csv
├── risk_map_2d.png
└── quadrant_summary.csv
```

### 12.5 成本情境分析

```text
output_cost_mc/
├── patient_lifetime_cost_mc.csv
├── mc_cost_per_patient.png
├── mc_population_total.png
├── mc_lorenz.png
└── cost_sensitivity_class_weight.csv
```

若成功合併含 `quadrant` 欄位的風險表，另產生：

```text
mc_cost_by_quadrant.png
```

---

## 13. 重現性

目前已實作：

- `SEED = 42`；
- sklearn 與 XGBoost 主要流程固定 random state；
- 分類與迴歸前處理封裝於 Pipeline；
- GroupKFold 依病患分組；
- OOF 評估；
- CGM 標準化只使用訓練段統計量；
- TensorFlow 設定 seed；
- Monte Carlo 使用固定 RNG seed。

仍需注意：

- 深度學習在不同硬體與 TensorFlow 版本下不一定完全 deterministic；
- requirements 尚未鎖定版本；
- 尚未在全新乾淨環境完成完整執行測試；
- feature importance、Odds Ratio 與 permutation importance 多數是全資料配適後的探索性結果；
- pooled LSTM 與逐監測 LSTM 的驗證設計不同。

---

## 14. 已知實作問題

1. `requirements.txt` 與五支主程式版本尚未鎖定。
2. pipeline 在病患去重模式下，`d_raw` 目前未同步去重。
3. risk 的 `transition_matrix()` 目前忽略 timestamp。
4. risk 的 Markov 起點固定為 `InRange`。
5. cost 的 fixed-age 模式對已超過 horizon 的病患仍保留至少 1 年。
6. cost 的部分「每人平均」與象限彙整仍使用全部紀錄。
7. `LIFE_EXPECTANCY` 尚未替換為正式官方生命表。

---

## 15. 研究限制

1. 臨床資料為橫斷面資料，不能直接推論未來發病或因果關係。
2. 樣本約 100 位病患，模型效能與重要性排序可能不穩定。
3. 部分生化變數缺失比例較高。
4. 分群結果不代表已驗證的臨床亞型。
5. 四象限不是臨床標準門檻。
6. 母體 AGP 與 Markov 以監測事件合併，較長監測可能取得較高權重。
7. 母體 Markov 尚未嚴格限制為 14–16 分鐘。
8. 個人 Markov 未排除跨缺測轉移。
9. LSTM 視窗可能跨越缺口。
10. 單一與逐監測 LSTM 的 validation 未明確保持時間順序。
11. pooled LSTM 仍使用測試監測自身歷史段建立標準化尺度。
12. 尚未使用外部資料驗證。
13. 風險 score 尚未完成正式機率校準。
14. 成本模型假設 Macro 與 Micro 可分開抽樣。
15. 成本模型未建模逐年病程進展與死亡。
16. CPI 分類口徑存在跨年度變化。
17. 年齡別餘命目前為暫定情境。
18. 成本輸出只能視為探索性情境分析。

---

## 16. 資料隱私

本 repository 不應包含：

- 原始病患臨床資料；
- 原始 CGM 檔案；
- 可識別病患的輸出表；
- 個人健康資訊。

請勿提交：

```text
Shanghai_T2DM_Summary.xlsx
Shanghai_T2DM/*.xls
Shanghai_T2DM/*.xlsx
data/clinical/*.xls
data/clinical/*.xlsx
data/cgm/*.xls
data/cgm/*.xlsx
```

提交前仍需人工確認：

- CSV 是否含 Patient Number；
- 圖表標題與檔名是否含病患編號；
- 文件與截圖是否可能重新識別病患；
- Git 歷史中是否曾提交原始資料。

---

## 17. 後續改進方向

- 建立 `project_schema.py`，集中管理欄位、mapping 與狀態門檻；
- 建立 `config.py`，統一輸入與輸出路徑；
- Markov 僅保留 14–16 分鐘有效轉移；
- 以真實最後狀態作為個人 Markov 起點；
- LSTM 在缺口處切段；
- 單一與逐監測 LSTM 使用 chronological validation；
- 以官方上海或中國生命表替換暫定餘命；
- 建立病患層級基準紀錄規則；
- 讓風險、象限與成本彙整使用相同分析單位；
- 附上完整 CPI 來源與連乘計算表；
- 進行 probability calibration；
- 加入自動測試與 GitHub Actions；
- 在乾淨環境完成五支程式的完整重現測試；
- 鎖定 Python 與套件版本；
- 若要建立真正的未來併發症與餘生成本模型，需加入縱向事件、逐年轉移與死亡資料。
