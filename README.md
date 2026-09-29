# 糖尿病風險分層與餘生醫療成本情境分析

輔仁統計資訊學系115年第26屆學生專題成果發表會第六組

本專題整合臨床併發症關聯分類、CGM 指標與 Markov／LSTM 分析、二維風險分層，以及探索性醫療成本模擬。App 為研究示範，未經臨床驗證，不能作為個別病患的診斷或治療依據。

## App

[開啟糖三臟研究示範 App](https://itzu-huang.github.io/Diabetes_Deterioration_Risk_Project/)

GitHub Pages 使用 `main` 分支的 `/docs`。`docs/index.html` 以相對路徑導向 App；Excel 解析使用同目錄內的 `vendor/xlsx.full.min.js`。使用者匯入的資料在瀏覽器內處理。

公開版保留研究模型參數與彙總結果，移除內嵌的個別病患檢驗測試資料及成本案例。頁面上的建置時驗證摘要不是每次開啟時重新執行的病患案例驗證。

## 目錄

```text
code/           分析與輸出驗證程式（15 支）
verification/   四層驗證入口、交付物核對腳本、豁免規則
docs/           GitHub Pages 首頁與 App
requirements.txt
.gitignore
```

報告、簡報與海報公開副本仍需完成去識別及版面驗收後才會加入。原始資料、病患層級輸出、凍結快照、執行日誌和本機環境資訊不隨公開 repo 發布。

## 安裝

來源交付環境使用 Python 3.13.9。`requirements.txt` 保留該環境的主要分析套件版本，另包含文件驗證所需的 `python-docx`、`python-pptx` 與 `Pillow`。

```sh
python -m venv .venv
# Windows PowerShell
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

## 資料與執行

來源為 ShanghaiT2DM 研究資料集，見 [figshare 資料頁](https://doi.org/10.6084/m9.figshare.21600933)。請依資料頁授權及說明自行取得。

將 `Shanghai_T2DM_Summary.xlsx` 與 `Shanghai_T2DM/` 放在 `code/`，再執行：

```sh
cd code
python run_all.py --with-sensitivity
```

完整執行包含深度學習，耗時依電腦而異。`--clean` 會清理程式輸出，只有需要從頭重跑且已保留必要結果時才使用。研究輸出保持在本機，不應加入版本控制。

## 驗證

```sh
python verification/verify_all.py --stage-dir code --deliverables-dir docs --scripts-dir verification
```

上述完整核對需要已驗收的報告／簡報／海報、分析輸出與獨立完整執行的凍結快照。只有程式與 App 的公開 checkout 不具備這些資料，缺少時應回報未執行或失敗，不可將此視為通過。

- 第 1 層：本次輸出與另一完整執行的凍結快照逐格比較。
- 第 2 層：交付文件及 App 中涵蓋的研究數字對照分析輸出。
- 第 3 層：數字是否能在輸出中找到，不代表每個出現位置都正確。
- 第 4 層：已涵蓋表格格位的四捨五入核對。

來源交付物曾通過四層檢查；公開副本的驗證狀態應另外記錄，不能沿用原件結果。跨作業系統的 XGBoost／LSTM 結果可能不同。

## 公開範圍

`.gitignore` 排除資料、逐筆輸出、模型檔、快照、壓縮包、環境秘密檔與執行日誌。忽略規則不會移除已追蹤檔案，因此提交前仍須檢查 staged diff。不要使用強制加入來繞過資料排除規則。
