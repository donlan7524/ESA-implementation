# Evolutionary Sampling Agent (ESA) for Expensive Problems

> 本專案為以下論文的實作與延伸研究：
>
> **"Evolutionary Sampling Agent for Expensive Problems"**
> *IEEE Transactions on Evolutionary Computation, 2023*

---

## 簡介

ESA核心想法是：在每一run中，由一個Agent根據當前的搜尋狀態，動態選擇最適合的Strategy來產生候選解，再交由Surrogate Model進行評估，以降低真實函數評估次數（NFE）。

本實作在原始論文基礎上進行了以下延伸：
- 新增多種代理模型後端支援（RBF-HPO、KNN、MLP、GP）
- 新增第 5 種優化策略：L-BFGS（a5）
- 支援以 L-BFGS 取代 SLS（a5 取代 a2）的策略組合實驗
- 實作多種強化學習 Agent （Thompson Sampling、UCB1）

---

## 專案結構

```text
ESA-implementation/
├── README.md
├── requirements.txt
├── main.py                   
├── experiment.py            
└── code/
    ├── RBF.py                
    ├── RBF_simple.py         
    ├── RBF_hpo.py            
    ├── KNN_surrogate.py      
    ├── MLP_hpo.py            
    ├── GP_surrogate.py      
    ├── Qagent.py            
    ├── TSagent.py           
    ├── UCBagent.py           
    ├── database.py          
    ├── kmeans.py             
    ├── brenchmarks/
    │   ├── __init__.py
    │   └── functions.py
    └── strategy/
        ├── base_Strategy.py
        ├── DE_strategy.py
        ├── SLS_strategy.py
        ├── crossover_strategy.py
        ├── TRLS_Strategy.py
        └── LBFGS_Strategy.py
```

---

## 方法簡述

### 整體流程

```
初始化（LHS 取樣）
       ↓
  訓練代理模型
       ↓
  Agent 選擇策略 aₜ
       ↓
  策略產生候選解
       ↓
  代理模型評估
       ↓
  更新資料庫 + Agent 學習
       ↓
  重複直到 NFE 達上限
```

### Q-learning 狀態設計

- 動作空間：n_actions = 4（標準）或 5（含 L-BFGS）
- 狀態空間：2 × n_actions 個狀態
  - 狀態轉移：`next_state = action × 2 + (1 if 改善成功 else 0)`
- 獎勵設計：成功改善 global best → `reward = 1 / n_evals`；否則 → 0

### 代理模型選項

| 模式 | 說明 |
| :--- | :--- |
| `simple` | 基礎 RBF，無超參最佳化 |
| `hpo` | RBF + LOOCV 自動選參 |
| `clustered_knn` | 以 K-means 分群後建立 KNN 模型 |
| `mlp` | 多層感知器 + Optuna 調參 |
| `gp` | 高斯過程 |

---

## 執行方式

### 安裝套件

```powershell
pip install -r requirements.txt
```

### 執行實驗

```powershell
python experiment.py
```

### 實驗設定

在 `experiment.py` 底部的設定區調整參數：

```python
os.environ["ESA_RBF_MODE"] = "hpo"            # 代理模型選擇
os.environ["ESA_USE_LBFGS"] = "False"         # True → 啟用5策略（加入 a5）
os.environ["ESA_REPLACE_A2_WITH_A5"] = "False" # True → 以 L-BFGS 取代 SLS
os.environ["ESA_QTABLE_ANALYSIS"] = "False"   # True → 實驗結束後輸出 Q-Table表格

run_progressive_experiments(
    dims=[30, 50, 100],       # 測試維度
    runs=10,                  # 每個配置的執行次數
    agent_types=["qlearning"] # Agent 類型：qlearning / ts / ucb
)
```

---

