import os
import sys
import time
import csv
import json
import concurrent.futures
import numpy as np
import pandas as pd

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# 引入本機的 ESA 模組
from database import Database
from strategy.DE_strategy import DE_Strategy
from strategy.SLS_strategy import SLS_Strategy
from strategy.TRLS_Strategy import TRLS_Strategy
from strategy.crossover_strategy import crossover_strategy
from strategy.LBFGS_Strategy import LBFGS_Strategy
# ==============================================================================
# ★ Colab 實驗設定區
# ==============================================================================
AGENT_TYPE = "qlearning"    # 'qlearning', 'ts', 'ucb', 'random', 'seq', 'alter'
USE_LBFGS = False          # True: 5策略版; False: 4策略原版
USE_MAHALANOBIS = True     # 測試函數與維度 (預設為 50D, 20 次，具體函數清單由底部 schedule 控制)
DIMS_TO_TEST = [50]
RUNS_PER_DIM = 20          # 論文設定 20 次

# 雲端硬碟備份路徑 (若在 Colab 運行且掛載了 Drive，會自動備份到該路徑)
GOOGLE_DRIVE_BACKUP_DIR = "/content/drive/MyDrive/ESA_Colab_Results"
# ==============================================================================

# 引入原版與 CEC 測試函數
import brenchmarks.functions as bf 
Benchmarks_config = {
    'Ellipsoid': {'func': bf.ellipsoid,'domain': [-5.12, 5.12], 'is_cec':False},
    'Rosenbrock': {'func': bf.rosenbrock,'domain': [-2.048,2.048], 'is_cec':False},
    'Ackley': {'func': bf.ackley, 'domain': [-32.768, 32.768], 'is_cec':False},
    'Griewank': {'func': bf.griewank, 'domain': [-600.0, 600.0], 'is_cec':False},
    'SRR': {'func': bf.srr, 'domain': [-5.0, 5.0], 'is_cec':True},
    'RHC1': {'func':bf.rhc1, 'domain': [-5.0, 5.0], 'is_cec':True},
    'RHC2': {'func':bf.rhc2, 'domain': [-5.0, 5.0], 'is_cec':True}
}

def make_agent(agent_type, n_actions, rng):
    if agent_type == "qlearning":
        from Qagent import Qlearning
        agent = Qlearning(alpha=0.1, gamma=0.9, T=1.0, rng=rng)
        agent.qtable = np.full((8, n_actions), 1.0 / n_actions)
        return agent
    elif agent_type == "ts":
        from TSagent import DiscountedTS
        return DiscountedTS(n_actions=n_actions, decay=0.98, rng=rng)
    elif agent_type == "ucb":
        from UCBagent import UCB1Selector
        return UCB1Selector(n_actions=n_actions, c=0.5, rng=rng)
    return None

def run_experiment(func_name, dim, seed):
    try:
        #設定初始參數
        MaxFEs = 1000
        init_samples = 150 if dim == 100 else 100
        rng = np.random.default_rng(seed)
        
        # 動態對照
        config = Benchmarks_config[func_name]
        
        # CEC函數初始設定方式不同
        if config['is_cec']:
            objective_func = config['func'](dim)
        else:
            objective_func = config['func']

        #設定上下界
        lb_val, ub_val = config['domain']
        bounds = np.array([[lb_val,ub_val]]*dim)

        #初始化資料庫
        DB = Database(dim, bounds)
        fes = DB.lhs(init_samples, objective_func)
        global_best_x, global_best_y = DB.getbest()
        global_best_y = float(global_best_y)

        # 根據全域開關動態裝載策略
        if USE_LBFGS:
            strategies = [  DE_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng),
                            SLS_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng),
                            crossover_strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng) ,
                            TRLS_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng),
                            LBFGS_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng)]
            n_actions = 5
        else:
            strategies = [  DE_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng),
                            SLS_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng),
                            crossover_strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng) ,
                            TRLS_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng)]
            n_actions = 4

        # 根據全域變數 AGENT_TYPE 導入對應的 agent
        Agent = make_agent(AGENT_TYPE, n_actions, rng)

        # 取得初始狀態
        if Agent is not None and hasattr(Agent, "get_initial_qtable"):
            current_state = Agent.get_initial_qtable()
        else:
            current_state = None

        action_count = np.zeros(len(strategies), dtype=int)   # 統計各策略被選次數
        success_count = np.zeros(len(strategies), dtype=int)
        history = [global_best_y] * fes
        history_improvements = {i: [] for i in range(n_actions)}

        stagnation_counter = 0
        stagnation_limit = 10
        step_count = 0

        #開始迴圈
        while fes < MaxFEs:
            # 選擇動作
            if Agent is not None:
                action = Agent.select(current_state) if hasattr(Agent, "select") else int(Agent.select_action(current_state))
                # 停滯保護路由
                if stagnation_counter >= stagnation_limit:
                    action = 1
                    stagnation_counter = 0
            else:
                action = 0

            selected_strategy = strategies[action]

            # 紀錄action被選擇次數與每次改善的幅度
            action_count[action] += 1
            D_new = selected_strategy.strategy(DB, objective_func)

            is_success = False
            n_evals = 0
            for x_new, y_new in D_new:
                if fes >= MaxFEs:
                    break
                DB.add_sample(x_new, float(y_new))
                fes += 1
                n_evals += 1
                if float(y_new) < global_best_y:
                    is_success = True
                    global_best_y = float(y_new)
            
            history.extend([global_best_y]*n_evals)

            if is_success:
                stagnation_counter = 0
                # 紀錄改善幅度 (前一次最優與這一次最優之差)
                if len(history) > n_evals + 1:
                    prev_best = history[-n_evals - 1]
                    improvement = prev_best - global_best_y
                else:
                    improvement = 0.0
                history_improvements[action].append(improvement)
            else:
                stagnation_counter += 1
                
            if Agent is not None:
                reward = Agent.compute_reward(is_success, n_evals)
                if hasattr(Agent, "next_state"):
                    next_state = Agent.next_state(action, is_success)
                    Agent.q_update(current_state, action, reward, next_state)
                    current_state = next_state
                else:
                    Agent.update(current_state, action, reward, None)
            
            step_count += 1
            
        return {'func': func_name, 'dim': dim, 'seed': seed, 'best_y': global_best_y, 'error': None}
    except Exception as e:
        return {'func': func_name, 'dim': dim, 'seed': seed, 'best_y': None, 'error': str(e)}

if __name__ == '__main__':
    # 固定為 Qlearning 決策智能
    AGENT_TYPE = "qlearning"
    
    schedule = [
        ("clustered_knn", ['Griewank', 'SRR', 'RHC1', 'RHC2']),
        ("mlp", ['Ellipsoid', 'Rosenbrock', 'Ackley', 'Griewank', 'SRR', 'RHC1', 'RHC2'])
    ]
    
    for mode, funcs_list in schedule:
        # 設定當前與子進程繼承的代理模型模式
        os.environ["ESA_RBF_MODE"] = mode
        
        # 建立任務清單
        tasks = []
        for func in funcs_list:
            for dim in DIMS_TO_TEST:
                for run in range(RUNS_PER_DIM):
                    tasks.append((func, dim, run))
                    
        total_tasks = len(tasks)
        print("\n" + "="*50)
        print(f"Colab 平行化實驗啟動 (總任務數: {total_tasks})")
        print(f"Surrogate Mode: {mode.upper()} | Agent: {AGENT_TYPE.upper()} | 策略數: {5 if USE_LBFGS else 4}")
        print("====================================================")
        
        # 嘗試引入 tqdm 輸出漂亮進度條
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False
            
        start_time = time.time()
        results = []
        
        # 使用所有可用 CPU 核心進行平行運算
        max_workers = os.cpu_count()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_experiment, f, d, s): (f, d, s) for f, d, s in tasks}
            
            # 根據是否有 tqdm 進行不同的迭代列印
            if has_tqdm:
                for future in tqdm(concurrent.futures.as_completed(futures), total=total_tasks, desc=f"Surrogate [{mode.upper()}] 進度"):
                    results.append(future.result())
            else:
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
                        
        end_time = time.time()
        elapsed = (end_time - start_time) / 60
        print(f"Surrogate [{mode.upper()}] 實驗完成！耗時: {elapsed:.2f} 分鐘")
        
        # 存檔處理
        if results:
            df = pd.DataFrame(results)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"colab_esa_results_{timestamp}_{AGENT_TYPE}_S_{mode}_L{int(USE_LBFGS)}.csv"
            
            # 1. 本地存檔
            df.to_csv(filename, index=False, float_format='%.4e')
            print(f"檔案已儲存至 Colab 本地空間: {filename}")
            
            # 2. 自動備份到雲端硬碟 (Drive)
            drive_path = "/content/drive"
            if os.path.exists(drive_path):
                try:
                    os.makedirs(GOOGLE_DRIVE_BACKUP_DIR, exist_ok=True)
                    backup_filepath = os.path.join(GOOGLE_DRIVE_BACKUP_DIR, filename)
                    df.to_csv(backup_filepath, index=False, float_format='%.4e')
                    print(f"雲端硬碟備份成功！已同步至: {backup_filepath}")
                except Exception as e:
                    print(f"雲端硬碟備份失敗: {e}")
            else:
                print("提示: 未偵測到雲端硬碟掛載，若需備份請先在 Colab 執行:")
                print("   from google.colab import drive")
                print("   drive.mount('/content/drive')")