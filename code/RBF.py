import math
import numpy as np

def distance(x1,x2):
    X1 = np.sum(x1 * x1, axis=1)[:, None]
    X2 = np.sum(x2 * x2, axis=1)[None, :]
    return np.maximum(X1 + X2 - 2.0 * (x1 @ x2.T), 0.0)
 

def caculate_Dmax(d2):
    Dmax = float(np.sqrt(d2.max()))
    return Dmax
            

# RBF => 擬合非線性問題
# {xi,yi} : x為解題參數之列表([[]]),y為fl(x)評估之分數列表([])
# d 為 問題參數dim , N 為 訓練資料筆數
class RBF():
    def __init__(self):
        #防止值為0
        self.eps = 1e-8
        
    # 找出需要參數    
    def fit(self,X : np.array,y : np.array) -> float:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        assert X.ndim == 2 and len(X) == len(y)
        X, idx = np.unique(X, axis=0, return_index=True)
        y = y[idx]
        N, d = X.shape
        d2 = distance(X, X)
        Dmax = float(np.sqrt(d2.max()))
        self.beta = max(Dmax * (d * N) ** (-1.0 / d), 1e-12)
        Phi = self._phi(d2)
        self.w = np.linalg.solve(Phi + self.eps * np.eye(N), y)
        self.X = X
        return self
    
    def predict(self,X_query):
        assert self.w is not None, "call fit() first"
        Xq = np.atleast_2d(np.asarray(X_query, dtype=float))
        pred = self._phi(self._sq_dists(Xq, self.X)) @ self.w
        return pred[0] if np.asarray(X_query).ndim == 1 else pred
    
    #創建 matrix Phi 
    def create_Phi(self,d2):
        return np.exp(-d2 / self.beta)  