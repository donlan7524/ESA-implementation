# Evolutionary Sampling Agent (ESA) for Expensive Problems

本專案為 IEEE Transactions on Evolutionary Computation (2023) 論文 “Evolutionary Sampling Agent for Expensive Problems” 的實作。


## 專案結構

本專案目前主要檔案如下：

```text
ESA-implementation/
├── README.md
├── code/
│   ├── run_experiments_parallel.py
│   ├── RBF.py
│   ├── RBF_hpo.py
│   ├── RBF_simple.py
│   ├── KNN_surrogate.py
│   ├── MLP_hpo.py
│   ├── GP_surrogate.py
│   ├── Qagent.py
│   ├── TSagent.py
│   ├── UCBagent.py
│   ├── database.py
│   ├── brenchmarks/
│   │   ├── __init__.py
│   │   └── functions.py
│   └── strategy/
│       ├── base_Strategy.py
│       ├── crossover_strategy.py
│       ├── DE_strategy.py
│       ├── LBFGS_Strategy.py
│       ├── SLS_strategy.py
│       └── TRLS_Strategy.py
└──
```



## 主要檔案說明

- `code/run_experiments_parallel.py`
  - 實驗入口
  - 配置 surrogate mode、agent list、dimension、runs
  - 收集結果並輸出到 `results.csv` 及 `experiment_logs/*.csv`
- `code/RBF.py`
  - 根據 `ESA_RBF_MODE` 動態選擇代理實作：
    - `simple` -> `RBF_simple.RBF`
    - `hpo` -> `RBF_hpo.RBF`
    - `clustered_knn` -> `KNN_surrogate.ClusteredKNN`
    - `mlp` -> `MLP_hpo.MLP_HPO`
    - `gp` -> `GP_surrogate.GP`
- `code/GP_surrogate.py`
  - Gaussian Process (高斯過程) 代理模型
- `code/Qagent.py`
  - Q-learning agent 的實作
- `code/TSagent.py`
  - Thompson Sampling agent
- `code/UCBagent.py`
  - UCB1 agent
- `code/database.py`
  - 儲存歷史樣本與 LHS 初始點生成
- `code/strategy/`
  - 四種優化策略：`DE`, `SLS`, `TRLS`, `crossover`, `LBFGS`
- `code/brenchmarks/functions.py`
  - benchmark 測試函數

## 執行方式

1. 進入 `code` 目錄：

```powershell
cd "c:\Users\Diego\Downloads\論文實作協助專案\new_clone\ESA-implementation\code"
```

2. 不知道


