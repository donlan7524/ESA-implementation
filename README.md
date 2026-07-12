# Evolutionary Sampling Agent (ESA) for Expensive Problems

> 本專案為以下論文的實作與延伸研究：
>
> **"Evolutionary Sampling Agent for Expensive Problems"**
> *IEEE Transactions on Evolutionary Computation, 2023*

---

## 簡介

ESA核心想法是：在每一run中，由一個Agent根據當前的搜尋狀態，動態選擇最適合的Strategy來產生候選解，再交由Surrogate Model進行評估，以降低真實函數評估次數（NFE）。

本實作在原始論文基礎上進行了以下延伸：
- 新增 RBF-HPO 代理模型（RBF + LOOCV 自動超參最佳化）
- 新增第 5 種優化策略：L-BFGS（a5）
- 支援以 L-BFGS 取代 SLS（a5 取代 a2）的策略組合
- 實作 Agent 並進行比較（Q-learning、Thompson Sampling、UCB1）

---

## 專案結構

```text
ESA-implementation/
├── README.md
├── CODE_ARCHITECTURE.md
├── requirements.txt
├── main.py
├── experiment.py
├── results.csv
├── experiment_logs/
├── experiment result and analysis/
│   ├── images/
│   │   ├── esa_result.png
│   │   ├── esa_qtable.png
│   │   ├── a5.png
│   │   └── ...
│   ├── 實驗結果.md
│   ├── 優化方向總覽.md
│   ├── 放棄的優化總覽.md
│   ├── RBF_hpo評估報告.md
│   ├── SLS 與 L-BFGS 策略比較.md
│   ├── TS_UCB_10seed完整矩陣報告.md
│   ├── 多群體策略失敗分析報告.md
│   ├── 馬氏距離實驗報告.md
│   └── ESA_收斂曲線.pdf
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

> [!TIP]
> 關於本專案的程式設計架構、核心演算法優化循環流程與各關鍵模組的實作細節，請參閱：[程式架構說明 (CODE_ARCHITECTURE.md)](CODE_ARCHITECTURE.md)

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

### 優化策略（Actions）

| 代號 | 策略 | 類型 | 說明 |
| :---: | :--- | :---: | :--- |
| a1 | DE | 全域探索 | 以差分向量擾動產生候選解，維持族群多樣性 |
| a2 | SLS | 局部利用 | 在當前最佳解附近隨機擾動，精細化局部區域 |
| a3 | Crossover | 重組 | 混合歷史樣本中的優良個體，產生新候選解 |
| a4 | TRLS| 局部利用 | 在信賴域範圍內建立局部代理模型並求最小值 |
| a5 | L-BFGS | 局部利用（選用） | 利用代理模型梯度資訊快速收斂，可替換 a2 |

### Q-learning 狀態設計

- 狀態空間：$2 \times n\_actions$ 個狀態
  - 狀態轉移：`next_state = action × 2 + (1 if 改善成功 else 0)`
- 獎勵設計：成功改善 global best → `reward = 1 / n_evals`；否則 → 0


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

或使用專案入口：

```powershell
python main.py
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
    agent_types=["qlearning"], # Agent 類型：qlearning / ts / ucb
    funcs=None,               # 指定測試函數，None 表示跑全部；例如 ["ellipsoid", "ackley"]
)
```

---

## 說明文件導覽

所有實驗分析報告位於 `experiment result and analysis/` 目錄下：

| 文件 | 內容摘要 |
| :--- | :--- |
| [實驗結果.md](<experiment%20result%20and%20analysis/實驗結果.md>) | 實驗結果截圖對照（Agent 比較、代理模型比較） |
| [優化方向總覽.md](experiment%20result%20and%20analysis/優化方向總覽.md) | 所有嘗試過的改動方向與當前採用狀態整理 |
| [放棄的優化總覽.md](experiment%20result%20and%20analysis/放棄的優化總覽.md) | 放棄的 5 項延伸：詳細動機、數據、放棄原因 |
| [RBF_hpo評估報告.md](experiment%20result%20and%20analysis/RBF_hpo評估報告.md) | RBF-HPO 作為主力模型的評估結果 |
| [SLS 與 L-BFGS 策略比較.md](<experiment%20result%20and%20analysis/SLS%20與%20L-BFGS%20策略比較.md>) | a2（SLS）vs a5（L-BFGS）策略替換實驗 |
| [TS_UCB_10seed完整矩陣報告.md](experiment%20result%20and%20analysis/TS_UCB_10seed完整矩陣報告.md) | Thompson Sampling / UCB1 與 Q-learning 橫向比較 |
| [多群體策略失敗分析報告.md](experiment%20result%20and%20analysis/多群體策略失敗分析報告.md) | MultiPop 三版迭代的失敗分析 |
| [馬氏距離實驗報告.md](experiment%20result%20and%20analysis/馬氏距離實驗報告.md) | 馬氏距離 RBF 的實驗結果 |
| [ESA_收斂曲線.pdf](experiment%20result%20and%20analysis/ESA_收斂曲線.pdf) | 各配置的收斂曲線圖 |
