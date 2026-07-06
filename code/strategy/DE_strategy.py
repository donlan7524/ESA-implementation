import numpy as np
from strategy.base_Strategy import Strategy
from RBF import  RBF


class DE_Strategy(Strategy):
    def __init__(self, lb, ub, rng, n=50, g_max=300, F=0.5, Cr=0.9):
        super().__init__(lb, ub, rng)  
        self.n = n                       # 族群/候選數
        self.g_max = g_max               # 全域模型資料上限
        self.F = F                       # DE 縮放因子
        self.Cr = Cr    
        
    def strategy(self,DB,f) :
        '''
        找出最好的50個歷史的解，
        利用fg打分找出最好的子代
        對最好的子代進行真實環境運算

        Args:
            DB: Database, the database containing historical samples and their fitness values
            f: callable, the real fitness function to evaluate new samples

        Returns:
            list of tuples, each containing a new sample and its evaluated fitness value
        '''
        #族群
        P, _ = DB.get_nbest(self.n)
        n,d = P.shape
        
        #建立全域代理
        g = min(len(DB),self.g_max)
        Xg, yg = DB.get_nbest(g)
        model = RBF().fit(Xg,yg)
        
        #產生突變及交配
        trials = self.gen_next(P)
        #確保解落在合法空間
        trials = self.clip(trials)
        
        #找出模擬分數最高子代
        pred = model.predict(trials)
        xc = trials[int(np.argmin(pred))]
 
        #真實評估
        D_new = [(xc, float(f(xc)))]
        return D_new
        
    def gen_next(self,P):
        """
        DE/rand/1 變異與二項式交叉產生新一代向量
        變異公式： vi = x_r1 + F*(x_r2 - x_r3)
        交叉公式： uij = vij if rand(j) <= Cr else xij

        Args:
            P: np.array, shape (n, d), current population
        Returns:
            np.array, shape (n, d), new population after mutation and crossover
        """

        
        """  """
        V = np.empty_like(P)
        n,d = P.shape
        for i in range(n):
            # 從 P 取 3 個相異索引
            r = self.rng.choice(np.delete(np.arange(n), i), size=3, replace=False)
            V[i] = P[r[0]] + self.F * (P[r[1]] - P[r[2]])
        # 二項式交叉
        # uij = vij if rand(j) <= Cr else xij
        mask = self.rng.random((n, d)) <= self.Cr
        jrand = self.rng.integers(0, d, size=n)
        mask[np.arange(n), jrand] = True     # 保證至少一維來自 V
        return np.where(mask, V, P)
            
        
    