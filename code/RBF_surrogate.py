import math
import numpy as np
import scipy.linalg as spla

def distance(x1,x2):
    X1 = np.sum(x1 * x1, axis=1)[:, None]
    X2 = np.sum(x2 * x2, axis=1)[None, :]
    return np.maximum(X1 + X2 - 2.0 * (x1 @ x2.T), 0.0)
            

# RBF => 擬合非線性問題
# {xi,yi} : x為解題參數之列表([[]]),y為fl(x)評估之分數列表([])
# d 為 問題參數dim , N 為 訓練資料筆數
class RBF():
    def __init__(self):
        #防止值為0
        self.eps = 1e-8
        self.y_mean = 0.0
        self.y_std = 1.0
        self.x_mean = 0.0
        self.W = None            # whitening 矩陣，使 ||(x1-x2)@W||^2 = 馬氏距離平方

    # 找出需要參數
    def fit(self,X : np.array,y : np.array) -> float:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        assert X.ndim == 2 and len(X) == len(y)

        X, idx = np.unique(X, axis=0, return_index=True)
        y = y[idx]
        N, d = X.shape

        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) + 1e-12
        y_scaled = (y - self.y_mean) / self.y_std

        # 估計樣本協方差並做 shrinkage 正則化：
        # N 相對 d 越小，樣本協方差本身的估計誤差越大，
        # 因此收縮，收縮強度隨 d/N 提高。
        self.x_mean = np.mean(X, axis=0)
        Xc = X - self.x_mean
        S = (Xc.T @ Xc) / max(N - 1, 1)
        trace_avg = float(np.trace(S)) / d
        if trace_avg <= 0:
            trace_avg = 1.0
        shrink = np.clip(d / N, 0.1, 0.9)
        S_reg = (1.0 - shrink) * S + shrink * trace_avg * np.eye(d)

        # 特徵分解得到 whitening 矩陣 W：Sigma^{-1} = W @ W.T
        eigval, eigvec = np.linalg.eigh(S_reg)
        eigval = np.maximum(eigval, 1e-12)
        self.W = eigvec @ np.diag(1.0 / np.sqrt(eigval))

        Xw = Xc @ self.W

        d2 = distance(Xw, Xw)
        Dmax = float(np.sqrt(d2.max()))
        self.beta = max(Dmax * (d * N) ** (-1.0 / d), 1e-12)

        Phi = self.create_Phi(d2)

        # 使用 scipy 的正定矩陣專用求解器，或以 lstsq 提高穩健性
        try:
            # 假設對稱正定，使用 assumes_a='pos' (基於 Cholesky 分解)
            self.w = spla.solve(Phi + self.eps * np.eye(N), y_scaled, assume_a='pos')
        except np.linalg.LinAlgError:
            # 若發生奇異矩陣錯誤，退回使用穩健的最小平方法
            self.w, _, _, _ = np.linalg.lstsq(Phi + self.eps * np.eye(N), y_scaled, rcond=None)

        self.X = Xw
        return self

    def predict(self,X_query):
        assert self.w is not None, "call fit() first"
        Xq = np.atleast_2d(np.asarray(X_query, dtype=float))
        Xqw = (Xq - self.x_mean) @ self.W

        pred_scaled = self.create_Phi(distance(Xqw, self.X)) @ self.w
        pred = pred_scaled * self.y_std + self.y_mean

        return pred[0] if np.asarray(X_query).ndim == 1 else pred
    
    #創建 matrix Phi 
    def create_Phi(self,d2):
        return np.exp(-d2 / self.beta ** 2)  
    
    def cubic(self,d2):
        return d2 ** 1.5
         