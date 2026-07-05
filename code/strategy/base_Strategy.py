import numpy as np

#四個策略都吃DB和function ，輸出D_new
#都要從 DB 取最佳若干筆
class Strategy():
    def __init__(self, lb, ub, rng):
        """
        lb, ub : 下界 / 上界 
        rng    : np.random.Generator, 所有策略共用同一個
        """
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.rng = rng                 
 
    def strategy(self, DB, f):   
        #不應該出現     
        raise NotImplementedError
 
    def clip(self, X):
        #將候選者拉回搜尋空間
        return np.clip(X, self.lb, self.ub)


