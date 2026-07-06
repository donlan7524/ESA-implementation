import numpy as np
from database import Database
from brenchmarks.functions import sphere
from strategy.DE_strategy import DE_Strategy

d = 20
MaxFEs = 1000
init_samples = 100
bounds = np.array([[-100, 100]]*d)
rng = np.random.default_rng(42)
history = []

DB = Database(d, bounds)
fes = DB.lhs(init_samples, sphere)
best_x, best_y = DB.getbest()
print("Initial best y:", best_y)
strategy = DE_Strategy(lb = bounds[:, 0], ub = bounds[:, 1], rng = rng)


while fes < MaxFEs:
    D_new = strategy.strategy(DB, sphere)
    
    for x_new, y_new in D_new:
        old_best_x, old_best_y = DB.getbest()
        
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