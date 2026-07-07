import numpy as np
from database import Database
import brenchmarks.functions as bf 
from strategy.DE_strategy import DE_Strategy
from strategy.SLS_strategy import SLS_Strategy
from strategy.TRLS_Strategy import TRLS_Strategy
from strategy.crossover_strategy import crossover_strategy

#設置維度
dim = 100
MaxFEs = 1000
init_samples = 100
bounds = np.array([[-100, 100]]*dim)
rng = np.random.default_rng(42)
history = []

DB = Database(dim, bounds)
fes = DB.lhs(init_samples, bf.shifted_sphere)
global_best_x, global_best_y = DB.getbest()

# 將a1 - a4 放入陣列中方便後面呼叫
strategies = [  DE_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng), 
                SLS_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng),
                crossover_strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng) ,
                TRLS_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng)]

print("Initial best y:", global_best_y)


while fes < MaxFEs:
    D_new = strategies[0].strategy(DB, bf.shifted_sphere)
    
    for x_new, y_new in D_new:
        
        DB.add_sample(x_new, y_new)
        fes += 1
        
        best_x, best_y = DB.getbest()
        history.append(float(best_y))
    
best_x, best_y = DB.getbest()

print("Final best y:", best_y)
print("Final best x:", best_x)
print("Final best x norm:", np.linalg.norm(best_x))

print("History length:", len(history))
print("First history:", history[0])
print("Last history:", history[-1])