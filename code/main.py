import numpy as np
import brenchmarks.functions as bf 

from Qagent import Qlearning
from database import Database
from strategy.DE_strategy import DE_Strategy
from strategy.SLS_strategy import SLS_Strategy
from strategy.TRLS_Strategy import TRLS_Strategy
from strategy.crossover_strategy import crossover_strategy


#基本參數的設定
dim = 100
MaxFEs = 1000
init_samples = 100
bounds = np.array([[-100, 100]]*dim)
rng = np.random.default_rng(42)

#初始化資料庫
DB = Database(dim, bounds)
fes = DB.lhs(init_samples, bf.sphere)
global_best_x, global_best_y = DB.getbest()
history = [global_best_y]

# 將a1 - a4 放入陣列中方便後面呼叫
strategies = [  DE_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng), 
                SLS_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng),
                crossover_strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng) ,
                TRLS_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng)]

# 導入Qlearning agent
init_T = 1.0
min_T= 0.1
Agent = Qlearning(alpha=0.1, gamma=0.9, T=init_T, rng=rng)
current_state = Agent.get_initial_qtable()

action_count = np.zeros(len(strategies), dtype=int)   # 統計各策略被選次數

print(f"Initial best y: {global_best_y}")

while fes < MaxFEs: 
    #透過current_state決定下個動作
    progess_ratio = fes / MaxFEs
    next_T = max(min_T, init_T * (1.0 - progess_ratio))
    
    action = Agent.select_action(current_state)
    selected_strategy = strategies[action]
    action_count[action]+=1
    
    D_new = selected_strategy.strategy(DB, bf.sphere)
    
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
        
    #假如找到更好，reward = 1 反之為 0
    reward = Agent.compute_reward(is_success,n_evals)
    
    #尋找下個狀態並更新q_table
    next_state = Agent.next_state(action, is_success)
    Agent.q_update(current_state, action, reward, next_state)
    current_state = next_state
    Agent.update_T(next_T)
    
best_x, best_y = DB.getbest()

print(f"Final best y: {best_y}")
print("Final Q-table:\n", np.round(Agent.qtable, 3))