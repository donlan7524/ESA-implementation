import numpy as np
from strategy.base_Strategy import Strategy
from scipy.optimize import differential_evolution
from RBF import RBF

class SLS_Strategy(Strategy):
    """
    a2 Surrogate Local Search Strategy
    流程：
    1. 從 DB 中找i個最佳解
    2. 建立 local RBF surrogate
    3. 透過最好 data 的 min/max 計算 local bounds
    4. 透過 DE 尋找最佳解
    5. 實際計算最佳解 fitness
    """
    def __init__(self, lb, ub, rng, l_best=None, pop_size=50, generations=50, F=(0.5,1), Cr=0.7, min_ratio=1e-6):
        super().__init__(lb, ub, rng)
        d = len(lb)
        
        self.l_best = l_best if l_best is not None else min(25 + d, 60)
        self.pop_size = pop_size
        self.generations = generations
        self.F = F
        self.Cr = Cr
        self.min_ratio = min_ratio
        print(self.l_best)
        
    def strategy(self, DB, f):
        """
        step 1 and 2 
        """
        l = min(self.l_best, len(DB))
        xl, yl = DB.get_nbest(l)
        model = RBF().fit(xl,yl)
        """
        step 3 : 
        """
        lb_local = np.min(xl, axis=0)
        ub_local = np.max(xl, axis=0)
        lb_local, ub_local = self.fix_local_bounds(lb_local, ub_local)

        """
        step 4:
        """
        xc = self.de_minimize(model, lb_local, ub_local)
        D_new = [(xc, float(f(xc)))]
        return D_new
    
    def fix_local_bounds(self, lb_local, ub_local):
        """
        確保 local bounds 不超過全域邊界，同時local bounds的寬度不小於min_width避免發生坍塌
        """
        lb_local = np.maximum(lb_local, self.lb)
        ub_local = np.minimum(ub_local, self.ub)
        global_width = self.ub - self.lb
        min_width = global_width * self.min_ratio
        
        for i in range(len(lb_local)):
            width = ub_local[i] - lb_local[i]
            if width < min_width[i]:
                center = (lb_local[i] + ub_local[i]) / 2
                half_min_width = min_width[i] / 2
                
                lb_local[i] = center - half_min_width
                ub_local[i] = center + half_min_width
                
                lb_local[i] = max(lb_local[i], self.lb[i]) # 確保更新後不超過原範圍
                ub_local[i] = min(ub_local[i], self.ub[i])
                
        return lb_local, ub_local
    
    def de_minimize(self, model, lb_local, up_local):
        """
        使用 scipy 中的 differential_evolution 來最小化 surrogate model
        """
        lb_local = np.asarray(lb_local)
        up_local = np.asarray(up_local)
        scipy_bounds = list(zip(lb_local, up_local))
        
        def surrogate_objective(x):
            y_pred = model.predict(x)
            return float(np.asarray(y_pred).ravel()[0])
        
        d = len(lb_local) 
        
        init_pop = self.rng.uniform(lb_local, up_local, size=(self.pop_size, d))
        result = differential_evolution(func=surrogate_objective, bounds=scipy_bounds,
                strategy="rand1bin",init=init_pop, maxiter=self.generations,
                mutation=self.F, recombination=self.Cr,polish=False, rng=self.rng)
        
        return np.copy(result.x)
        