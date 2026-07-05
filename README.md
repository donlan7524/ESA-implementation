# Evolutionary Sampling Agent (ESA) for Expensive Problems  
本專案為 IEEE Transactions on Evolutionary Computation (2023) 論文 *“Evolutionary Sampling Agent for Expensive Problems”* 的實作。  


---

## 專案目錄可能的結構

```text
ESA.../
├── code/
│   ├── __init__.py       # not sure
│   ├── database.py       # 管理歷史數據與 LHS 初始化
│   ├── RBF.py            # 高斯 RBF 代理模型
│   ├── optimizers.py     # JADE
│   ├── strategies.py     # a1-a4的實作
│   └── agent.py          # Q-learning
│   └── benchmarks/       # still not sure
│       ├── __init__.py
│       └── functions.py      
├── main.py                  
├── requirements.txt          # 依賴套件
└── README.md                 # 本說明文件
```

---

## 檔案函式說明

### database.py
程式內說明待補，順帶說明使用單純一個class存因為效率可能會比較快
* `lhs(self, n_samples, real_fitness_func)`: 邊界內進行lhs
* `add_sample(x, y)`: 新增單個歷史data
* `get_best()`: 獲取當前最優解及值
* `def get_nbest(self, n):` 獲取前 $n$ 個最優點
* `nearest_point(self, center, count)`: 篩選距離 `center` 最近的 `count` 個點
* `get_samples_range(self, lb, ub)`: 篩選落在區間 `[lb, ub]` 內的所有點

### RBF.py
pass

## 實驗需求
1.  最大評估次數: $MaxFEs = 1000$。
2.  LHS 初始化樣本數:
    *   30D / 50D 問題：$100$ 次評估。
    *   100D 問題：$150$ 次評估。
3.  測試函數: Sphere, Ellipsoid, Rosenbrock, Ackley, Griewank, Rastrigin。
4.  運行次數: 每個函數在各維度下需獨立運行 20 次，並統計結束時最適應度的平均值與標準差。
5.  a1-a4的消融實驗
