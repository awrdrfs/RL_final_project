# 📈 深度強化學習股票交易系統 (RL Stock Trading System)

本專案是一個基於深度強化學習（Deep Reinforcement Learning, DRL）的股票交易決策系統。專案中實現了不同的強化學習演算法（如 DQN 與 Q-Transformer），並設計了專屬的股票交易環境（`TradingEnv`），透過歷史股價數據來訓練代理人（Agent）進行買賣決策，以最大化資產回報率。

---

## 📁 專案目錄結構

```text
RL_final_project/
│
├── 📄 yfinance_crawler.py               # 股票數據爬蟲腳本
├── 📄 dqn_trading.py                    # 經典 DQN 交易訓練（包含 Replay Buffer）
├── 📄 dqn_trading_no_buffer.py           # DQN 交易訓練（無 Replay Buffer，線上立即更新）
├── 📄 qtransformer_trading_no_buffer..py # Q-Transformer 交易訓練（自迴歸離散動作模型）
├── 📄 draw_line_chart.py                # 訓練與驗證結果視覺化繪圖腳本
│
├── 📂 dataset/                          # 股票數據集目錄
│   ├── 📂 train/                        # 訓練用數據（CSV 格式，含多個個股及組合）
│   └── 📂 validTest/                    # 驗證與測試用數據（CSV 格式）
│
├── 📂 train_result/                     # 訓練結果與產出目錄
│   ├── 📂 log/                          # 訓練與驗證的 CSV 日誌數據
│   │   ├── 📂 DQN_woReplayBuffer_vt/    # 無 Replay Buffer DQN 的日誌
│   │   └── 📂 Qtransformer_woReplayBuffer_vt/ # Q-Transformer 的日誌
│   └── 📂 model/                        # 訓練完成的模型權重檔 (.pt / .pt 狀態字典)
│       ├── 📂 DQN_woReplayBuffer_vt/    # 無 Replay Buffer DQN 模型權重
│       └── 📂 Qtrans_woReplayBuffer_vt/ # Q-Transformer 模型權重
│
└── 📂 chat_log_with_AI /                # 與 AI 討論的對話記錄（PDF 格式，設計與 Debug 參考）
```

---

## 📂 資料夾功能詳細介紹

### 1. `dataset/` (數據集)
儲存從 Yahoo Finance 爬取的股票歷史交易數據。
* **`train/`**：包含用於訓練的模型數據。
* **`validTest/`**：包含驗證與測試模型泛化能力的數據。
* **命名規則**：
  * `*.csv`：包含盤前與盤後交易時段的完整數據（以 1 小時為單位）。
  * `*_wo_prepost.csv`：**不包含**盤前與盤後交易，僅包含常規交易時段的數據（因為盤前盤後交易量低且波動大，移除後有助於 RL 環境的穩定收斂）。
  * `total_stock.csv`：多隻股票合併後的綜合數據。

### 2. `train_result/` (訓練產出)
存放所有模型訓練過程中的數據與權重。
* **`log/`**：記錄每個 Episode 的總回報（Total Reward）、資產價值（Portfolio Value）及探索率（Epsilon），格式為 CSV，便於後續分析。
* **`model/`**：存放訓練中表現最好（Best Model）以及最後一個 Episode（End Model）的 PyTorch 模型權重（`.pt` 檔案）。

### 3. `chat_log_with_AI /` (AI 討論日誌)
收集了專案開發過程中與 AI (ChatGPT/Gemini) 討論的技術 PDF 記錄。內容涵蓋：
* 環境初始化 bug 調試（例如 `TradingEnv` 為 `None` 的處理）。
* 如何將 Transformer 結構融入強化學習（如 SAC 演算法中 Replay Buffer 學習長期依賴）。
* 數據格式處理與特徵工程等思路。

---

## 📄 Python 檔案功能詳細介紹

### 1. [yfinance_crawler.py](file:///Users/bradpan/Desktop/深度強化學習/RL_final_project/yfinance_crawler.py)
* **功能**：使用 `yfinance` 庫自動下載指定股票代碼（如 AAPL, NVDA, JPM, XOM, DIS）的歷史數據。
* **特性**：
  * 下載 1 小時（`interval="1h"`）精細度的數據。
  * 自動下載兩種類型：包含盤前盤後數據（`prepost=True`）與不含盤前盤後數據（`prepost=False`）。
  * 自動清除 `yfinance` 返回的多重索引標頭（`Ticker` 行），並將結果寫入 `dataset/` 下的對應路徑。

### 2. [dqn_trading.py](file:///Users/bradpan/Desktop/深度強化學習/RL_final_project/dqn_trading.py)
* **功能**：基於 PyTorch 實現**經典 DQN（Deep Q-Network）**股票交易算法。
* **關鍵元件**：
  * `TradingEnv`：自訂的交易環境（詳見下方說明）。
  * `ReplayBuffer`：經驗回放緩衝區，儲存歷史轉換以打破數據相關性，隨機抽樣進行批次更新（Batch Size = 64）。
  * `QNetwork`：全連接神經網絡，用於估算每個動作狀態的 Q 值。
* **運作機制**：採用 $\epsilon$-greedy 探索策略，逐步衰減 $\epsilon$，並使用雙網絡架構（DQN + Target Network）定期同步以穩定訓練。

### 3. [dqn_trading_no_buffer.py](file:///Users/bradpan/Desktop/深度強化學習/RL_final_project/dqn_trading_no_buffer.py)
* **功能**：移除經驗回放區（Replay Buffer）的 DQN 訓練變體。
* **運作機制**：
  * Agent 採用**線上更新（Online Learning）**，每執行一步就立即使用該單步經驗更新 Q 網絡。
  * 用於作為對照組，以便評估經驗回放在股票時間序列數據中對於收斂性與穩定性的影響。
  * 預設使用迪士尼（DIS）股票數據進行實驗。

### 4. [qtransformer_trading_no_buffer..py](file:///Users/bradpan/Desktop/深度強化學習/RL_final_project/qtransformer_trading_no_buffer..py)
* **功能**：參考 CoRL 2023 論文 *"Q-Transformer: Scalable Offline RL via Autoregressive Q-Functions"* 實現的高級模型。
* **核心技術**：
  * **動作離散化（Action Tokenizer）**：將連續的動作空間（0~1 的股票/現金持倉比例）均勻切分成多個區間（預設為 21 個 Action Bins），轉化為離散的類別。
  * **自迴歸解碼（Autoregressive Decoding）**：透過 Transformer 的 Causal Mask 機制，自迴歸地依序解碼每個動作維度（先決定股票權重，再根據其結果決定現金權重）。
  * **Dueling Head**：結合 Dueling DQN 架構，將輸出拆分為狀態價值（Value）與動作優勢（Advantage），藉此提升動作評估的準確度。
  * **時間差分目標（TD Target）**：在自迴歸解碼的多個步驟中，只有在最後一個動作解碼完成並與環境交互時才給予真實 Reward，前序步驟 Reward 設為 0。

### 5. [draw_line_chart.py](file:///Users/bradpan/Desktop/深度強化學習/RL_final_project/draw_line_chart.py)
* **功能**：讀取訓練產出的 CSV 日誌，並繪製對比圖表。
* **功能細節**：
  * 繪製並保存**驗證集損益比（Profit & Loss Ratio）**隨 Episode 變化的對比曲線。
  * 繪製並保存**訓練集總回報（Total Reward）**對比曲線。
  * 用以直觀評估「有經驗回放」與「無經驗回放」等不同設定下的效能差異。

---

## ⚙️ 交易環境介紹 (`TradingEnv`)

專案中的 `TradingEnv` 模擬了真實市場的交易流程，具有以下核心機制：

1. **狀態空間 (State / Observation)**:
   * 當前小時的 OHLCV（開盤價、最高價、最低價、收盤價、成交量）數據。
   * 目前持有股票數量。
   * 當前現金餘額。
2. **動作空間 (Action)**:
   * 輸出一個向量，經過 Softmax 轉換後映射成 **[股票配置比例, 現金配置比例]**（例如 `[0.7, 0.3]` 代表將總資產的 70% 配置於股票，30% 持有現金）。
3. **交易機制**:
   * 當配置比例需要買入時，考慮交易手續費率（預設為 0.3%），若現金不足，則以最大可購現金進行買入；若配置比例需要賣出，同樣扣除手續費。
4. **獎勵函數 (Reward)**:
   * 基礎獎勵為本期資產淨值相較於上一期的**回報率（Return %）**。
   * **破產/違規懲罰**：若 Agent 試圖執行超出規則的行為（如超額買入導致現金透支、超額賣出等），會被額外扣除懲罰分 `-1.0`，強迫模型學習資金流與庫存控管。

---

## 🚀 快速開始步驟

### 1. 安裝環境依賴
確保安裝以下 Python 庫：
```bash
pip install numpy pandas torch yfinance matplotlib
```

### 2. 下載股票數據
打開 `yfinance_crawler.py` 修改你想獲取的股票與日期，然後運行：
```bash
python yfinance_crawler.py
```

### 3. 開始訓練模型
可運行經典 DQN、無 Buffer DQN 或 Q-Transformer：
```bash
# 運行傳統 DQN
python dqn_trading.py

# 運行無經驗回放的 DQN
python dqn_trading_no_buffer.py

# 運行自迴歸 Q-Transformer
python qtransformer_trading_no_buffer..py
```

### 4. 繪製對比圖表
運行繪圖腳本可視化訓練成果：
```bash
python draw_line_chart.py
```
繪製出的比較圖將會儲存至 `train_result/image/` 資料夾中。
