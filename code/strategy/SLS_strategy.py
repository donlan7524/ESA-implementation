import numpy as np
from strategy.base_Strategy import Strategy
from scipy.optimize import differential_evolution
from RBF import RBF

class SLS_Strategy(Strategy):
    """
    a2 Surrogate Local Search Strategy (JADE Optimization Edition)
    流程：
    1. 從 DB 中找i個最佳解
    2. 建立 local RBF surrogate
    3. 透過最好 data 的 min/max 計算 local bounds
    4. 透過 JADE 尋找最佳解
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
        xl, yl = DB.get_nbest(min(self.l_best, len(DB)))
        model = RBF().fit(xl,yl)
        """
        step 3 : 
        """
        lb_local = np.min(xl, axis=0)
        ub_local = np.max(xl, axis=0)
        lb_local, ub_local = self.fix_local_bounds(lb_local, ub_local)

        """
        step 4: 使用 JADE 
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

    def jade_minimize(self, model, lb_local, ub_local):
        """
        
        """
        lb = np.asarray(lb_local)
        ub = np.asarray(ub_local)
        d = len(lb)

        pop_size = 100
        maxiter = 300

        #JADE 基本參數
        p = 0.5
        c = 0.1
        mu_cr = 0.5
        mu_f = 0.5
        archive = []

        pop = self.rng.uniform(lb, ub, (pop_size, d))
        pop = np.clip(pop, self.lb, self.ub)
        fitness = np.array(model.predict(pop)).flatten()

        for gen in range(maxiter):
            S_cr = []
            S_f = []

            cr = self.rng.normal(mu_cr, 0.1, pop_size)
            cr = np.clip(cr, 0, 1)

            f_dist = mu_f + 0.1 * np.tan(np.pi * (self.rng.random(pop_size) - 0.5))
            retry_mask = f_dist <= 0
            while np.any(retry_mask):
                num_retries = np.sum(retry_mask)
                f_dist[retry_mask] = mu_f + 0.1 * np.tan(np.pi * (self.rng.random(num_retries) - 0.5))
                retry_mask = f_dist <= 0
            f_dist = np.clip(f_dist, 0, 1)

            #找fitness前p%的
            sorted_index = np.argsort(fitness)
            p_best_index = sorted_index[:max(1, int(pop_size * p))]

            new_pop = np.zeros_like(pop)
            #建立一個包含「當前族群 + Archive」
            pop_archive = np.vstack((pop, np.array(archive))) if len(archive) > 0 else pop


            pbest_choices = self.rng.choice(p_best_index, size=pop_size)
            x_pbest = pop[pbest_choices]

            #隨機選擇 r1
            r1_targets = np.empty(pop_size, dtype=int)
            #$v_i = x_i + F_i \cdot (x_{pbest} - x_i) + F_i \cdot (x_{r1} - \tilde{x}_{r2})$ 目標公式
            for i in range(pop_size):
                r1_targets[i] = self.rng.choice(np.delete(np.arange(pop_size), i))
            x_r1 = pop[r1_targets]

            #隨機選擇 r2
            r2_targets = self.rng.choice(len(pop_archive), size=pop_size)
            x_r2 = pop_archive[r2_targets]

            v = pop + f_dist[:, None] * (x_pbest - pop) + f_dist[:, None] * (x_r1 - x_r2)

            # 5. crossover 開始
            rand_matrix = self.rng.random((pop_size, d))
            j_rand = self.rng.integers(0, d, size=pop_size)
            mask = rand_matrix <= cr[:, None]
            mask[np.arange(pop_size), j_rand] = True

            u = np.where(mask, v, pop)
            u = np.clip(u, lb, ub)
            new_pop = u

            new_fitness = np.array(model.predict(new_pop)).flatten()

            #Archive 更新
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
                keep_idx = self.rng.choice(len(archive), pop_size, replace=False)
                archive = [archive[idx] for idx in keep_idx]

            # 更新自適應均值
            if len(S_cr) > 0:
                mu_cr = (1 - c) * mu_cr + c * np.mean(S_cr)
                sf_arr = np.array(S_f)
                mu_f = (1 - c) * mu_f + c * (np.sum(sf_arr**2) / np.sum(sf_arr))

        best_idx = np.argmin(fitness)
        return np.copy(pop[best_idx])