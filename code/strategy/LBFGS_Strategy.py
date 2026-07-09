import numpy as np
from strategy.base_Strategy import Strategy
from scipy.optimize import minimize
from RBF import RBF

class LBFGS_Strategy(Strategy):
    """
    a5: 基於梯度的代理模型局部尋優 (L-BFGS-B)
    流程：
    1. 取出歷史資料中最優秀的 l_best 筆資料建立 Local RBF。
    2. 以當前「歷史最佳解」為起點 (x0)。
    3. 利用 scipy.optimize.minimize (L-BFGS-B) 沿著梯度滑向代理模型的最低點。
    4. 進行真實評估。
    """
    def __init__(self, lb, ub, rng, l_best=None):
        super().__init__(lb, ub, rng)
        d = len(lb)
        self.l_best = l_best if l_best is not None else max(2*d, 10)
        
    def strategy(self, DB, f):
        """
        Step 1: 建立局部代理模型與設定起點
        """
        x0, f_best = DB.getbest()
        x_local, y_local = DB.nearest_point(x0, self.l_best)
        model = RBF().fit(x_local,y_local)
        global_width = self.ub - self.lb
        
        def surrogate_objective(x):
            pred = model.predict(np.atleast_2d(x))
            return float(np.asarray(pred).ravel()[0])

        margin = x_local.max(axis=0) - x_local.min(axis=0)
        margin = np.maximum(margin, global_width*0.05)
        lb_local = np.maximum(x_local.min(axis=0) - 0.5 * margin, self.lb)
        ub_local = np.minimum(x_local.max(axis=0) + 0.5 * margin, self.ub)
        bounds = list(zip(lb_local,ub_local))
        
        result = minimize(
            fun=surrogate_objective,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-9, 'eps': 1e-10}
        )
        
        xc = self.clip(result.x)
        D_new = [(xc, float(f(xc)))]
        return D_new
        
        
        
        