# 程式架構說明

本文件詳細說明本專案的程式設計架構、演算法執行流程，以及各關鍵模組的實作細節。

---

## 專案程式架構總覽

整個專案的原始碼設計分為三個核心層次，以確保模組之間的低耦合度與高擴充性：

- Control & Execution
  - `main.py`：單次實驗與除錯的快速入口。
  - `experiment.py`：批次實驗的核心執行器，負責管理實驗參數、載入測試函數、記錄實驗日誌並輸出 Q-Table。

- Surrogate & Strategies
  - `code/RBF.py`：代理模型分流器，根據環境變數 `ESA_RBF_MODE` 動態載入並路由至指定的代理模型實作。
  - `code/RBF_simple.py`：最基礎的 RBF 模型，使用固定的正則化與核寬度係數，無自動調參。
  - `code/RBF_hpo.py`：本專案的主力代理模型，整合 Rippa 演算法與留一交叉驗證網格搜尋，並加入二次多項式尾項以改善外插外推的能力。
  - `code/KNN_surrogate.py`：已放棄的代理模型。使用 K-Means 將樣本分群並在各局部區域訓練 KNN 迴歸器，因預測速度慢且小樣本下精準度不穩而停用。
  - `code/MLP_hpo.py`：已放棄的代理模型。使用多層感知器並以 Optuna 尋找層數與學習率等超參數，在 1000 NFE 以內的小數據場景下容易過擬合且調參成本太高。
  - `code/GP_surrogate.py`：實驗性代理模型，基於高斯過程（Gaussian Process）實作。
  - `code/strategy/base_Strategy.py`：策略基底類別，定義所有優化策略的標準輸入輸出介面，確保其能動態接受資料庫與代理模型。
  - `code/strategy/DE_strategy.py`（a1 策略）：Differential Evolution優化策略，用於全域探索，以防止陷入局部最佳解。
  - `code/strategy/SLS_strategy.py`（a2 策略）：Stochastic Local Search，在目前最佳解的鄰域內生成候選點進行局部收斂。
  - `code/strategy/crossover_strategy.py`（a3 策略）：交叉重組策略，利用歷史優良樣本的交叉特徵產生新的優化候選點。
  - `code/strategy/TRLS_Strategy.py`（a4 策略）：Trust-Region Local Search，在縮小後的信賴半徑區域內最佳化局部代理模型。
  - `code/strategy/LBFGS_Strategy.py`（a5 策略）：選用的擬牛頓優化策略，利用代理模型估計數值梯度進行高效率局部收斂，可用於直接替換 SLS（a2）。

- Decision & State
  - `code/Qagent.py`：Q-learning 決策 Agent，負責策略選擇與 Q-Table 更新。
  - `code/TSagent.py` / `code/UCBagent.py`：無狀態的 Bandit 決策 Agent。
  - `code/database.py`：實驗資料庫，負責管理歷史評估樣本、記錄目前最佳解，並提供LHS初始化。

---

## 核心演算法執行流程

演算法的核心執行流程實作於 `experiment.py` 的 `run_esa` 函數中。單次實驗的生命週期如下：

- 初始化階段
  - 呼叫 `Database` 透過 LHS 在搜尋空間中產生初始點。
  - 對初始點進行真實函數評估，記錄初始的 Global Best。

- 迭代搜尋階段（重複執行直到真實評估次數 NFE 達到上限）
  - 步驟一：代理模型訓練
    - 讀取資料庫中的歷史樣本。
    - 使用 `RBF_hpo` 訓練代理模型，以逼近真實函數的地形。
  - 步驟二：Agent 選擇策略
    - Agent 根據目前的搜尋狀態（State），透過 $\epsilon$-greedy 或是機率取樣選擇本輪要採用的策略（Action $a_t$）。
  - 步驟三：策略生成候選點
    - 被選中的策略以當前最佳解為中心生成一定數量的候選點池。
  - 步驟四：代理模型評估與篩選
    - 使用步驟一訓練好的代理模型對候選點池進行快速預測。
    - 挑選出代理模型預測值最好的點，作為真實評估的候選點。
  - 步驟五：真實評估與更新
    - 對篩選出的候選點進行真實函數評估。
    - 將新樣本寫入資料庫。
  - 步驟六：學習與回饋
    - 檢查新評估點是否成功改善了 Global Best。
    - 根據改善結果計算Reward，並更新Agent的決策參數（如 Q-Table）。

---

## 關鍵模組分析與實作細節

### Q-learning 決策引擎 (`code/Qagent.py`)

- 狀態空間
  - 狀態轉移公式實作為：`state = last_action * 2 + (1 if last_improved else 0)`。
  - 此設計使狀態總數固定為 $2 \times n\_actions$。

- 動作空間動態切換
  - 建構子讀取策略列表長度以動態初始化 Q-Table 大小為 `(n_actions * 2, n_actions)`。
  - 當啟用 L-BFGS 策略（$n\_actions=5$）或進行策略替換時，Q-Table 的維度會自動適應調整，避免索引越界。

- 自適應獎勵設計
  - 改善成功時，獎勵值計算為 $Reward = 1 / n\_evals$，其中 $n\_evals$ 為目前的真實評估次數。
  - 改善失敗時，獎勵值固定為 $0$。

### RBF-HPO 代理模型 (`code/RBF_hpo.py`)

- 超參數自動調優 (Rippa 演算法)
  - 核心基底函數（RBF）採用高斯核：$\phi(r) = \exp(-\beta r^2)$。
  - 實作 Rippa 演算法，利用留一交叉驗證誤差公式，在一維對數網格中自動搜尋最佳的核寬度 $\beta$ 與嶺正則化係數 $\lambda$。

- Quadratic Polynomial Tail
  - 在標準 RBF 方程式中加入多項式尾項 $P(x) = \sum c_i x_i + \sum d_i x_i^2$。

### 策略池實作 (`code/strategy/`)

- L-BFGS 近似梯度優化 (`code/strategy/LBFGS_Strategy.py`)
  - 作為 a5 策略，在昂貴優化問題中，真實函數通常無法直接求導。
  - 實作中利用 `scipy.optimize.minimize` 的 L-BFGS-B 演算法，並將「RBF 代理模型」作為評估函數。
  - L-BFGS 運作時所需的梯度資訊由 RBF 代理模型在候選點上的數值差分近似提供，預想在不耗費任何真實評估預算的前提下，實現局部收斂。
