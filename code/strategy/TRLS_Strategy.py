import numpy as np
from strategy.base_Strategy import Strategy
from scipy.optimize import differential_evolution
from RBF import RBF
'''

輸入: DB, 真實評估 f
輸出: D_new (3 筆)

取 DB 最佳m(100)筆 → DBm
用 DBm 算外層包絡盒 [lb, ub] (式 7,8, 和 a2 同一招)
kmax = 3, k = 0
設定初始信賴域與初始半徑 Δ0
迴圈 k < kmax:
   取[信賴域 且 外層盒]內的資料建局部 RBF 的虛擬評分
   在信賴域內最小化 f̂ → 候選 xc
   真實評估 f(xc), 加入 D_new  ← 每輪迭代耗 1 FE
   若 f(xc) < f(xbest): 更新 xbest
   算信賴比 ρk, 更新半徑 Δk+1
   f. k += 1
6. 回傳 D_new (共 3 筆)
(注意)消耗之FE為3，選擇時需考慮預算(其他三個方法預算皆為1)
'''

class TRLS_Strategy(Strategy):
    def __init__(self, lb, ub, rng, m=100, kmax=3, xi=2.0,
                    pop_size=50, generations=50, F=0.5, Cr=0.9,
                    eps_denom=1e-12, poly_tail_min_ratio=1.5):
            super().__init__(lb, ub, rng)
            self.m = m                    # 外層盒資料量 (100)
            self.kmax = kmax              # 信賴域迭代次數 = 消耗 FE 数 (3)
            self.xi = xi                  # ρ>=0.75 时半徑放大係數
            self.pop_size = pop_size      # 内部 DE 設定 
            self.generations = generations
            self.F = F
            self.Cr = Cr
            self.eps_denom = eps_denom    # ρ 分母保護閥值
            self.poly_tail_min_ratio = poly_tail_min_ratio
        
    def strategy(self, DB, f):   
        d = DB.d
        
        #外層包裹盒
        Xm,ym = DB.get_nbest(min(self.m,len(DB)))
        global_width = self.ub - self.lb
        margin = Xm.max(axis=0) - Xm.min(axis=0)
        margin = np.maximum(margin, 1e-5*global_width)

        lb_outer = np.maximum(Xm.min(axis=0) - 0.5 * margin, self.lb)
        ub_outer = np.minimum(Xm.max(axis=0) + 0.5 * margin, self.ub)
        
        #初始半徑
        xbest,f_best = DB.getbest()
        X_nb,y_nb = DB.nearest_point(xbest,5*d)
        x_min_resp = X_nb[int(np.argmin(y_nb))]   #界內y最小點
        x_max_resp = X_nb[int(np.argmax(y_nb))]   #界內y最大點
        init_delta = 0.5 * np.linalg.norm(x_min_resp - x_max_resp) #步伐為1半[ymax到ymin]
        if init_delta <= 0:                             # 鄰居和自己重合
            init_delta = 0.05 * np.mean(self.ub - self.lb)
          
        delta = init_delta  
        #創建資料池  
        X_pool, y_pool = np.copy(Xm), np.copy(ym)
        
        D_new = []
        for k in range(self.kmax):
            # 信賴域 x_best + [-delta,delta] 且 外層盒
            tr_lb = np.maximum(xbest-delta,lb_outer)
            tr_ub = np.minimum(xbest+delta,ub_outer)
            span = tr_ub - tr_lb
            collapsed = span < 1e-10 #預防崩潰
            tr_lb[collapsed] = np.maximum(tr_lb[collapsed] - 1e-6, self.lb[collapsed])
            tr_ub[collapsed] = np.minimum(tr_ub[collapsed] + 1e-6, self.ub[collapsed])
            
            #信賴域內的RBF
            in_tr = np.all((X_pool >= tr_lb) & (X_pool <= tr_ub), axis=1)
            X_tr, y_tr = X_pool[in_tr], y_pool[in_tr]
            
            min_required = int(np.ceil(self.poly_tail_min_ratio * (2 * d + 1))) + 10
            if len(X_tr) < min_required:            # 資料不足(含只達到 d+1 的邊緣情況)，找鄰近點補足
                X_tr, y_tr = DB.nearest_point(xbest, max(min_required, 2 * d))
            """
            if len(X_tr) < d + 1:                  # 資料不足，找鄰近點
                X_tr, y_tr = DB.nearest_point(xbest, max(d + 1, 2 * d))"""
            model = RBF().fit(X_tr, y_tr)
            
            #代理評估
            xc = self.de_minimize(model, tr_lb, tr_ub)
            xc = self.clip(xc)
            
            #真實評估
            yc = float(f(xc))
            D_new.append((np.copy(xc), yc))
            X_pool = np.vstack([X_pool, xc])       # 新解加入資料池
            y_pool = np.append(y_pool, yc)
            
            #信賴比
            pred_best = float(np.asarray(model.predict(xbest)).ravel()[0])
            pred_c = float(np.asarray(model.predict(xc)).ravel()[0])
            denom = pred_best - pred_c
            if abs(denom) > self.eps_denom:
                rho = (f_best - yc) / denom 
            else:
                rho = -1.0  #分母不可信，失敗                    
    
            #更新 xbest, 再更新半徑
            if yc < f_best:
                xbest, f_best = np.copy(xc), yc
                
            min_delta = 1e-5
            if rho <= 0.25:
                delta *= 0.25
            elif rho >= 0.75:
                delta *= self.xi
                
            if delta < min_delta:
                delta = init_delta / 2.0
            # 0.25 < rho < 0.75: delta 不變
 
        return D_new 
    
    def de_minimize(self, model, lb_local, ub_local):
        lb_local = np.asarray(lb_local)
        ub_local = np.asarray(ub_local)
        #優化器，同a2之DE
        def surrogate_batch(x):
            return model.predict(np.atleast_2d(x.T))
 
        d = len(lb_local)
        result = differential_evolution(
            func=surrogate_batch,
            bounds=list(zip(lb_local, ub_local)),
            strategy="rand1bin",
            popsize=max(1, int(np.ceil(self.pop_size / d))),
            maxiter=self.generations,
            mutation=self.F, recombination=self.Cr,
            polish=False, rng=self.rng,
            vectorized=True,
            updating='deferred')
        return np.copy(result.x)