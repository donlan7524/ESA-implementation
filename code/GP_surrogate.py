import numpy as np
import scipy.linalg as spla

class GP:
    def __init__(self, length_score=1.0, signal_var=1.0, noise_var=1e-4):
        self.l = length_score
        self.signal_var = signal_var
        self.noise_var = noise_var
        
        self.x_train = None
        self.y_train = None
        self.L = None
        self.alpha = None
        
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        
    def _distance(self, x1, x2):
        X1 = np.sum(x1 * x1, axis=1)[:, None]
        X2 = np.sum(x2 * x2, axis=1)[None, :]
        return np.maximum(X1 + X2 - 2.0 * (x1 @ x2.T), 0.0)
    
    def _kernel(self, x1, x2):
        d2 = self._distance(x1,x2)
        return self.signal_var * np.exp(-0.5 * d2 / (self.l ** 2))
    
    # 對外
    
    def fit(self, x, y):
        x = np.atleast_2d(x)
        y = np.atleast_1d(y).flatten()
        
        x, unique_idx = np.unique(x, axis=0, return_index=True)
        y = y[unique_idx]
        
        # 檢查過濾後資料是否太少
        if len(x) < 2:
            raise ValueError("過濾重複點後資料不足以建立模型")
        
        # 標準化
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0) + 1e-8
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8
        
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std
        
        self.x_train = x_norm
        self.y_train = y_norm
        
        # 計算歷史點之間的共變異數(相似度)矩陣 K
        K = self._kernel(x_norm,x_norm)
        
        #加入雜訊
        K += self.noise_var * np.eye(len(x_norm))
        
        try:
            self.L = spla.cholesky(K, lower=True)
        except spla.LinAlgError:
            K += 1e-3 * np.eye(len(x_norm))
            self.L = spla.cholesky(K, lower=True)
            
        temp = spla.solve_triangular(self.L, y_norm, lower=True)
        self.alpha = spla.solve_triangular(self.L.T, temp, lower=False)
        return self
    
    def predict(self, x_query, return_std=False):
        assert self.L is not None, "請先呼叫fit()"
        
        x_query = np.atleast_2d(x_query)
        x_q_norm = (x_query - self.x_mean) / self.x_std
        
        K_star = self._kernel(self.x_train,x_q_norm)
        mu_norm = K_star.T @ self.alpha
        
        mu = mu_norm * self.y_std +self.y_mean
        if not return_std:
            return mu