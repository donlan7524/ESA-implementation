import numpy as np
import optuna
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.exceptions import ConvergenceWarning

#完全靜音sklearn與Optuna的所有輸出，避免干擾TUI排版
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
optuna.logging.set_verbosity(optuna.logging.CRITICAL)

class MLP_HPO:
    """
    每隔特定次數的擬合呼叫才執行一次超參數搜尋
    中間的擬合呼叫直接沿用上次找到的最佳參數並進行快速重新擬合
    """

    _cached_params = None    #上次Optuna找到的最佳超參數
    _fit_call_count = 0      #全域fit呼叫計數器

    def __init__(self, n_trials=3, hpo_interval=5, *args, **kwargs):
        """
        Args:
        n_trials (int)：自動尋優的嘗試次數，降低次數以加速
        hpo_interval (int)：執行超參數尋優的擬合間隔次數
        """
        self.n_trials = n_trials
        self.hpo_interval = hpo_interval
        self.model = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def _build_model(self, params):
        #根據超參數字典建立MLP模型
        hidden_depth = params["hidden_depth"]
        if hidden_depth == 1:
            hidden_layer_sizes = (params["hidden_unit1"],)
        else:
            hidden_layer_sizes = (params["hidden_unit1"], params["hidden_unit2"])
        return MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=params["activation"],
            solver="adam",
            alpha=params["alpha"],
            max_iter=150,
            random_state=42
        )

    def _default_params(self):
        #樣本過少時使用的保守超參數
        return {"hidden_depth": 1, "hidden_unit1": 10, "activation": "tanh", "alpha": 1e-3}

    def _run_optuna(self, X, y_scaled):
        #執行超參數搜尋，使用單次劃分驗證取代交叉驗證
        def objective(trial):
            hidden_depth = trial.suggest_int("hidden_depth", 1, 2)
            if hidden_depth == 1:
                hidden_layer_sizes = (trial.suggest_int("hidden_unit1", 5, 30),)
            else:
                hidden_layer_sizes = (
                    trial.suggest_int("hidden_unit1", 5, 30),
                    trial.suggest_int("hidden_unit2", 5, 20)
                )
            activation = trial.suggest_categorical("activation", ["tanh", "relu", "logistic"])
            alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)

            #用訓練集與驗證集劃分取代交叉驗證
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y_scaled, test_size=0.33, random_state=42
            )
            try:
                reg = MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    solver="adam",
                    alpha=alpha,
                    max_iter=150,
                    random_state=42
                )
                reg.fit(X_tr, y_tr)
                return float(mean_squared_error(y_val, reg.predict(X_val)))
            except Exception:
                return 1e6

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) + 1e-12
        y_scaled = (y - self.y_mean) / self.y_std

        #樣本太少時直接用預設超參數跳過尋優
        if len(X) <= 10:
            params = self._default_params()
            self.model = self._build_model(params)
            self.model.fit(X, y_scaled)
            return self

        #超參數尋優──
        MLP_HPO._fit_call_count += 1
        need_hpo = (
            MLP_HPO._cached_params is None or
            MLP_HPO._fit_call_count % self.hpo_interval == 1
        )

        if need_hpo:
            MLP_HPO._cached_params = self._run_optuna(X, y_scaled)

        #用快取或剛搜尋到的最佳超參數重新擬合全量資料
        self.model = self._build_model(MLP_HPO._cached_params)
        self.model.fit(X, y_scaled)
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        assert self.model is not None, "Call fit() first"
        return self.model.predict(X) * self.y_std + self.y_mean

    def predict_single(self, x):
        return self.predict(x)[0]
