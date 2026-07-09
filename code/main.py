import os
# ⚠️ 效能關鍵：必須在 import numpy 之前設定，關閉底層的自動多執行緒，
# 避免多進程同時呼叫 numpy 時引發嚴重的 CPU 執行緒衝突 (Thread Thrashing)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import brenchmarks.functions as bf 
import concurrent.futures
import time
import pandas as pd
import json
import os

from Qagent import Qlearning
from database import Database
from strategy.DE_strategy import DE_Strategy
from strategy.SLS_strategy import SLS_Strategy
from strategy.TRLS_Strategy import TRLS_Strategy
from strategy.crossover_strategy import crossover_strategy
from strategy.LBFGS_Strategy import LBFGS_Strategy


#定義每個要做的實驗種類
Benchmarks_config = {
    'Ellipsoid': {'func': bf.ellipsoid,'domain': [-5.12, 5.12], 'is_cec':False},
    'Rosenbrock': {'func': bf.rosenbrock,'domain': [-2.048,2.048], 'is_cec':False},
    'Ackley': {'func': bf.ackley, 'domain': [-32.768, 32.768], 'is_cec':False},
    'Griewank': {'func': bf.griewank, 'domain': [-600.0, 600.0], 'is_cec':False},
    'SRR': {'func': bf.srr, 'domain': [-5.0, 5.0], 'is_cec':True},
    'RHC1': {'func':bf.rhc1, 'domain': [-5.0, 5.0], 'is_cec':True},
    'RHC2': {'func':bf.rhc2, 'domain': [-5.0, 5.0], 'is_cec':True}
}

def run_experiment(func_name, dim, seed):
    try:
        #設定初始參數
        MaxFEs = 1000
        init_samples = 150 if dim == 100 else 100
        rng = np.random.default_rng(seed)
        
        config = Benchmarks_config[func_name]
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

        # 將a1 - a4 放入陣列中方便後面呼叫
        strategies = [  DE_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng), 
                        SLS_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng),
                        crossover_strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng) ,
                        TRLS_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng),
                        LBFGS_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng)]

        # 導入Qlearning agent
        Agent = Qlearning(alpha=0.1, gamma=0.9, T=1.0, rng=rng)
        current_state = Agent.get_initial_qtable()

        action_count = np.zeros(len(strategies), dtype=int)   # 統計各策略被選次數
        success_count = np.zeros(len(strategies), dtype=int)
        history = [global_best_y] * fes
        history_improvements = {0: [], 1: [], 2: [], 3: [], 4: []}
        
        stagnation_counter = 0
        stagnation_limit = 10
        #開始迴圈
        while fes < MaxFEs: 
            #透過current_state決定下個動作
            action = Agent.select_action(current_state)
            selected_strategy = strategies[action]
            
            if stagnation_counter >= stagnation_limit:
                action = 1
                stagnation_counter = 0
            
            #紀錄action被選擇次數與每次改善的幅度
            action_count[action]+=1
            current_best = DB.getbest()[1]
            
            D_new = selected_strategy.strategy(DB, objective_func)
            is_success = False
            
            n_evals = 0 #實際入帳FE
            
            for x_new, y_new in D_new:
                if fes>= MaxFEs:
                    break
                DB.add_sample(x_new, float(y_new))
                fes += 1
                n_evals += 1
                
                if float(y_new) < global_best_y:
                    is_success = True
                    global_best_y = float(y_new)
                    global_best_x = np.copy(x_new)
                
                history.append(global_best_y)                    
            
            if is_success:
                success_count[action]+=1
                new_best = DB.getbest()[1]
                improvement = float(current_best) - float(new_best)
                history_improvements[action].append(improvement)
            else:
                stagnation_counter += 1
                
            #假如找到更好，reward = 1 反之為 0
            reward = Agent.compute_reward(is_success,n_evals)
            
            #尋找下個狀態並更新q_table
            next_state = Agent.next_state(action, is_success)
            Agent.q_update(current_state, action, reward, next_state)
            current_state = next_state
       
        #回傳實驗資料
        return {
            'func': func_name,
            'dim': dim,
            'seed': seed,
            'best_y': global_best_y,
            'history': json.dumps(history),
            'action_count':action_count,
            'success_count':success_count,
            'history_improvments': json.dumps(history_improvements)
        }
    except Exception as e:
        return {'func': func_name, 'dim': dim, 'seed': seed, 'error': str(e)}

if __name__ == '__main__':
    funcs_to_test = ['Ellipsoid', 'Rosenbrock', 'Ackley', 'Griewank', 'SRR', 'RHC1', 'RHC2'] 
    dims_to_test = [30, 50]
    runs_per_dim = 1 # 論文設定每組跑 20 次
    
    task = []
    for func in funcs_to_test:
        for dim in dims_to_test:
            if dim == 100 and Benchmarks_config[func]['is_cec']:
                print(f"⚠️ 跳過 {func}-100D，因為 opfunu 套件不支援此維度。")
                continue
            for run in range(runs_per_dim):
                task.append((func,dim,run))
    total_task = len(task)
    print (f"準備開始執行 {total_task} 個執行任務")
    start_time = time.time()
    
    result = []
    
    #開啟多核心運算
    #若不想跑電腦核心數-1，請調整max_core
    max_core = max(1,os.cpu_count()-1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_core) as executer:
        futures = {executer.submit(run_experiment, f, d, s): (f, d, s) for f, d, s in task}
        
        complete = 0
        for future in concurrent.futures.as_completed(futures):
            complete += 1
            res = future.result()
            
            if 'error' in res:
                print(f"❌ [進度 {complete}/{total_task}] 任務失敗 ({res['func']}-{res['dim']}D, Seed {res['seed']}): {res['error']}")
            else:
                print(f"✅ [進度 {complete}/{total_task}] 完成 {res['func']}-{res['dim']}D (Seed {res['seed']}) -> 最終收斂: {res['best_y']:.4e}")
            
            result.append(res)

    end_time = time.time()
    print(f"實驗結束 總耗時:{(end_time - start_time) / 60:.2f} min")
    
    if result:
        df = pd.DataFrame(result)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"esa_benchmark_results_{timestamp}.csv"
        
        df.to_csv(filename, index=False, encoding='utf-8-sig', float_format='%.4e')
        print(f"📊 實驗結果已成功儲存至：{filename}")
    else:
        print("⚠️ 沒有產生任何結果，跳過 CSV 匯出。")
