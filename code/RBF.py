import warnings
import numpy as np
import scipy.linalg as spla

"""
原本RBF問題:
1.病態矩陣 解 加正則化，變成 Ridge/regularized RBF
2.Shape parameter ϵ 用 LOOCV 誤差最小化來選 ϵ
3.純 RBF 沒有 polynomial trend

"""

def distance(x1, x2):
    """成對距離平方矩陣 ||x1_i - x2_j||^2"""
    X1 = np.sum(x1 * x1, axis=1)[:, None]
    X2 = np.sum(x2 * x2, axis=1)[None, :]
    return np.maximum(X1 + X2 - 2.0 * (x1 @ x2.T), 0.0)



# RBF (含線性多項式尾項) => 擬合非線性問題，用來預測昂貴目標函數 fitness
# {xi,yi} : x為解題參數之列表([[]]),y為fl(x)評估之分數列表([])
# d 為 問題參數dim , N 為 訓練資料筆數
class RBF():
    def __init__(self, beta_candidates=None, lam_candidates=None, use_poly_tail=True,
                 poly_tail_min_ratio=1.2):
        """
        beta_candidates : shape parameter 候選倍率(乘上 Dmax_normalized)，用 LOOCV 挑最佳
        lam_candidates  : Tikhonov 正則化係數候選，用 LOOCV 挑最佳
        use_poly_tail   : 是否「嘗試」加入線性多項式尾項 (實際是否使用還要看樣本數是否足夠)
        poly_tail_min_ratio : 多項式尾項的啟動條件之一 啟動失敗則用原版RBF
        """
        if beta_candidates is None:
            beta_candidates = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
        if lam_candidates is None:
            lam_candidates = np.array([1e-10, 1e-8, 1e-6, 1e-4, 1e-2])

        self.beta_candidates = beta_candidates
        self.lam_candidates = lam_candidates
        self.use_poly_tail = use_poly_tail
        self.poly_tail_min_ratio = poly_tail_min_ratio

        self.eps = 1e-12
        self.y_mean = 0.0
        self.y_std = 1.0
        self.loocv_rmse = None
        self.poly_tail_active = False   # 這次 fit() 實際上有沒有用到多項式尾項
        self.ill_conditioned = False    # 是否曾偵測到病態警告(供除錯用)

    # ---------- 內部工具 ----------

    def _normalize_X(self, X):
        """線性映射到 [0,1]^d，讓不同 domain 尺度的問題使用同一套 beta/lambda 邏輯"""
        return (X - self.X_min) / (self.X_range + 1e-12)

    def create_Phi(self, d2, beta):
        return np.exp(-d2 / beta ** 2)

    def _build_system(self, Phi, d, N):
        """
        建立含多項式尾項的擴增線性系統：
        [ Phi   P ] [w]   [y]
        [ P^T   0 ] [c] = [0]
        其中 P = [1, x] (常數項 + 線性項)，並施加正交條件 sum(w)=0, sum(w*x)=0
        """
        P = np.hstack([np.ones((N, 1)), self.X])          # N x (d+1)
        top = np.hstack([Phi, P])
        bottom = np.hstack([P.T, np.zeros((d + 1, d + 1))])
        A = np.vstack([top, bottom])
        return A, P

    def _safe_solve_sym(self, A, rhs):
        """
        求解對稱(但非正定，saddle-point結構)的增廣系統，
        並偵測 LinAlgWarning(病態矩陣)；一旦偵測到，回傳 None 讓呼叫端知道要降級處理，
        而不是靜默吃下一個不可靠的解。
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", category=spla.LinAlgWarning)
            try:
                sol = spla.solve(A, rhs, assume_a='sym')
            except np.linalg.LinAlgError:
                return None
            if any(issubclass(w.category, spla.LinAlgWarning) for w in caught):
                self.ill_conditioned = True
                return None
        return sol

    def _solve(self, Phi, y_scaled, lam, d, N, use_poly_tail):
        """給定 beta 對應的 Phi 與 lambda，解出 w (與多項式係數 c，若啟用且數值穩健)"""
        Phi_reg = Phi + lam * np.eye(N)
        if use_poly_tail:
            A, P = self._build_system(Phi_reg, d, N)
            rhs = np.concatenate([y_scaled, np.zeros(d + 1)])
            sol = self._safe_solve_sym(A, rhs)
            if sol is None:
                # 增廣系統病態或奇異 -> 用最小平方法兜底，仍然嘗試給出合理解
                sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
            w = sol[:N]
            c = sol[N:]
            return w, c
        else:
            try:
                w = spla.solve(Phi_reg, y_scaled, assume_a='pos')
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(Phi_reg, y_scaled, rcond=None)
            return w, None

    def _loocv_rmse(self, Phi, y_scaled, lam, d, N, use_poly_tail):
        """
        用 Rippa's method 快速估計 LOOCV 誤差，避免真的重複解 N 次線性系統。
        若啟用多項式尾項，退化為對擴增矩陣直接做 leave-one-out（成本較高但 N 通常不大）。
        """
        Phi_reg = Phi + lam * np.eye(N)

        if not use_poly_tail:
            try:
                Phi_inv = np.linalg.inv(Phi_reg)
            except np.linalg.LinAlgError:
                return np.inf
            w = Phi_inv @ y_scaled
            diag = np.diag(Phi_inv)
            diag = np.where(np.abs(diag) < 1e-14, 1e-14, diag)
            errors = w / diag
            return float(np.sqrt(np.mean(errors ** 2)))
        else:
            A, P = self._build_system(Phi_reg, d, N)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", category=spla.LinAlgWarning)
                try:
                    Ainv = np.linalg.inv(A)
                except np.linalg.LinAlgError:
                    return np.inf
                if any(issubclass(w_.category, spla.LinAlgWarning) for w_ in caught):
                    # 這組 (beta, lambda) 在增廣系統下已經病態，直接淘汰，
                    # 不要讓病態解的假性低 LOOCV 誤差誤導超參數選擇
                    return np.inf
            rhs = np.concatenate([y_scaled, np.zeros(d + 1)])
            sol = Ainv @ rhs
            w = sol[:N]
            diag = np.diag(Ainv)[:N]
            diag = np.where(np.abs(diag) < 1e-14, 1e-14, diag)
            errors = w / diag
            return float(np.sqrt(np.mean(errors ** 2)))

    # --- 外部funcionts ---

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        assert X.ndim == 2 and len(X) == len(y)

        X, idx = np.unique(X, axis=0, return_index=True)
        y = y[idx]
        N, d = X.shape

        # 決定這次 fit() 是否有資格用多項式尾項
        # 當資料庫中資料不足時採用純RBF 否則容易出現奇異解
        self.poly_tail_active = self.use_poly_tail and (N >= self.poly_tail_min_ratio * (d + 1))
        self.ill_conditioned = False

        # 輸入正規化
        self.X_min = X.min(axis=0)
        self.X_max = X.max(axis=0)
        self.X_range = self.X_max - self.X_min
        X = self._normalize_X(X)
        self.X = X

        #輸出標準化
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) + 1e-12
        y_scaled = (y - self.y_mean) / self.y_std

        d2 = distance(X, X)
        Dmax = float(np.sqrt(d2.max())) if d2.max() > 0 else 1.0

        # 測試所以候選
        best_beta, best_lam, best_err = None, None, np.inf
        for beta_ratio in self.beta_candidates:
            beta = max(Dmax * beta_ratio, 1e-6)
            Phi = self.create_Phi(d2, beta)
            for lam in self.lam_candidates:
                err = self._loocv_rmse(Phi, y_scaled, lam, d, N, self.poly_tail_active)
                if err < best_err:
                    best_err, best_beta, best_lam = err, beta, lam

        # 假如候選解都不合適 退回原版RBF
        if self.poly_tail_active and not np.isfinite(best_err):
            self.poly_tail_active = False
            for beta_ratio in self.beta_candidates:
                beta = max(Dmax * beta_ratio, 1e-6)
                Phi = self.create_Phi(d2, beta)
                for lam in self.lam_candidates:
                    err = self._loocv_rmse(Phi, y_scaled, lam, d, N, False)
                    if err < best_err:
                        best_err, best_beta, best_lam = err, beta, lam

        # 若所有候選都失敗，退回保守預設值
        if best_beta is None:
            best_beta = max(Dmax * 0.3, 1e-6)
            best_lam = 1e-6

        self.beta = best_beta
        self.lam = best_lam
        self.loocv_rmse = best_err if np.isfinite(best_err) else None

        Phi = self.create_Phi(d2, self.beta)
        self.w, self.c = self._solve(Phi, y_scaled, self.lam, d, N, self.poly_tail_active)

        return self

    def predict(self, X_query):
        assert self.w is not None, "call fit() first"
        raw_ndim = np.asarray(X_query).ndim
        Xq = np.atleast_2d(np.asarray(X_query, dtype=float))
        Xq_norm = self._normalize_X(Xq)

        Phi_q = self.create_Phi(distance(Xq_norm, self.X), self.beta)
        pred_scaled = Phi_q @ self.w

        if self.poly_tail_active and self.c is not None:
            P_q = np.hstack([np.ones((Xq_norm.shape[0], 1)), Xq_norm])
            pred_scaled = pred_scaled + P_q @ self.c

        pred = pred_scaled * self.y_std + self.y_mean
        return pred[0] if raw_ndim == 1 else pred

    def cubic(self, d2):
        return d2 ** 1.5