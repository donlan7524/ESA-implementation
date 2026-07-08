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
    def __init__(self, lb, ub, rng, l_best=None, pop_size=10, generations=150, F=(0.5,1), Cr=0.7, min_ratio=1e-6):
        super().__init__(lb, ub, rng)
        d = len(lb)
        
        self.l_best = l_best if l_best is not None else min(25 + d, 60)
        self.pop_size = pop_size
        self.generations = generations
        self.F = F
        self.Cr = Cr
        self.min_ratio = min_ratio
        
    def strategy(self, DB, f):
        """
        step 1 and 2 
        """
        xl, yl = DB.get_nbest(self.l_best)
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
        xc = self.jade_minimize(model, lb_local, ub_local)
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
                strategy="rand1bin",popsize=self.pop_size, maxiter=self.generations,
                mutation=self.F, recombination=self.Cr,polish=False, rng=self.rng)
        return np.copy(result.x)
        
    def jade_minimize(self, model, lb_local, ub_local):
        lb = np.asarray(lb_local)
        ub = np.asarray(ub_local)
        d = len(lb)
        
        pop_size = 100
        maxiter = 300
        
        #設定JADE基本參數
        p = 0.5
        c = 0.1
        mu_cr = 0.5
        mu_f = 0.5
        archive=[]
        
        pop = self.rng.uniform(lb, ub, (pop_size, d))
        fitness = np.array(model.predict(pop)).flatten()
        
        for gen in range(maxiter):
            S_cr = []
            S_f= []
            
            cr = self.rng.normal(mu_cr, 0.1, pop_size) #生成CR
            cr = np.clip(cr, 0, 1) #大於1的話截斷在1
            
            #生成F
            f_dist = mu_f + 0.1* np.tan( np.pi * (self.rng.random(pop_size) - 0.5))
            
            #處理不合格的數值
            retry_mask = f_dist <= 0
            while np.any(retry_mask):
                num_retries = np.sum(retry_mask)
                f_dist[retry_mask] = mu_f + 0.1* np.tan( np.pi * (self.rng.random(num_retries) - 0.5))
                retry_mask = f_dist <= 0
                
            f_dist = np.clip(f_dist, 0, 1)#大於1的話截斷在1
            
            #找fitness前p%的
            sorted_index = np.argsort(fitness)
            p_best_index = sorted_index[:max(1, int(pop_size * p))]
            
            new_pop = np.zeros_like(pop)
            
            #建立一個包含「當前族群 + Archive」
            pop_archive = np.vstack((pop,np.array(archive))) if len(archive) > 0 else pop
            
            #$v_i = x_i + F_i \cdot (x_{pbest} - x_i) + F_i \cdot (x_{r1} - \tilde{x}_{r2})$ 目標公式
            for i in range(pop_size):
                #隨機從pbest中挑取一個
                x_pbest = pop[self.rng.choice(p_best_index)]
                
                #去除自己並從pop中隨機挑取一個
                r1_candidate = np.delete((np.arange(pop_size)),i)
                r1_target = self.rng.choice(r1_candidate)
                
                #選擇r1
                x_r1 = pop[r1_target]
                #選擇r2
                r2_target = self.rng.choice(len(pop_archive))
                x_r2 = pop_archive[r2_target]
                
                #計算 mutation
                v = pop[i] + f_dist[i] * (x_pbest - pop[i]) +  f_dist[i] * (x_r1 - x_r2)
                
                # crossover 開始
                j_rand = self.rng.integers(0,d)
                mask = self.rng.random(d) <= cr[i]
                mask[j_rand] = True
                
                u = np.where(mask, v, pop[i])
                u = np.clip(u, lb, ub)
                new_pop[i] = u
            
            new_fitness = np.array(model.predict(new_pop)).flatten()    
            
            for i in range(pop_size):
                #假如變異後結果好過原本的
                if new_fitness[i] < fitness[i]:
                    #存取當下cr 與 f
                    S_cr.append(cr[i])
                    S_f.append(f_dist[i])
                    
                    #並將優秀父代放入archive中並用新的取代其原本位置
                    archive.append(pop[i].copy())
                    pop[i] = new_pop[i]
                    fitness[i] = new_fitness[i]
            
            #確保archive的長度不超過pop_size
            if len(archive) > pop_size:
                keep_idx = self.rng.choice(len(archive), pop_size, replace = False)
                archive = [archive[idx] for idx in keep_idx]
             
            if len(S_cr) > 0:
                mu_cr = (1 - c) * mu_cr + c * np.mean(S_cr)
                sf_arr = np.array(S_f)
                mu_f = (1 - c) * mu_f + c * (np.sum(sf_arr**2) / np.sum(sf_arr))
                   
            best_idx = np.argmin(fitness)
        return np.copy(pop[best_idx])
            
            
            