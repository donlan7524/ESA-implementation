import numpy as np
from scipy.spatial.distance import cdist
from kmeans import k_means, predict_cluster

class ClusteredKNN:
    def __init__(self, n_clusters=5, k_neighbors=3, weights="distance", select="k means++", *args, **kwargs):
        """
        拿了資料探勘的期末抱個來改的結合Kmeans的KNN模型

        
        """
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.weights = weights
        self.select = select
        
        self.X_train = None
        self.y_train = None
        self.centers = None
        self.X_by_cluster = {}
        self.y_by_cluster = {}
        self.global_mean = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.X_train = X
        self.y_train = y
        self.global_mean = float(np.mean(y))
        
        num_unique = len(np.unique(X, axis=0))
        effective_clusters = max(1, min(self.n_clusters, num_unique))
            
        try:
            cluster, centers, _ = k_means(X, effective_clusters, self.select)
            self.centers = centers
            
            if self.centers is not None and len(self.centers) > 0:
                labels, _ = predict_cluster(X, self.centers)
            else:
                labels = np.zeros(len(X), dtype=int)
            
            self.X_by_cluster = {}
            self.y_by_cluster = {}
            for c in range(effective_clusters):
                mask = (labels == c)
                if np.any(mask):
                    self.X_by_cluster[c] = X[mask]
                    self.y_by_cluster[c] = y[mask]
                else:
                    self.X_by_cluster[c] = X
                    self.y_by_cluster[c] = y
        except Exception:
            self.centers = None
            self.X_by_cluster = {0: X}
            self.y_by_cluster = {0: y}
            
        return self

    def predict(self, X_query):
        """
        單點分支預測與向量化批量預測
        """
        X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
        assert self.X_train is not None, "Call fit() first"
        
        n_query = len(X_query)
        
        #單點分支預測
        if n_query == 1:
            x = X_query[0]
            if self.centers is not None and len(self.centers) > 0:
                #找出最鄰近的中心索引
                c = np.argmin(cdist(X_query, self.centers)[0])
            else:
                c = 0
            
            X_local = self.X_by_cluster.get(c, self.X_train)
            y_local = self.y_by_cluster.get(c, self.y_train)
            
            #距離計算
            dists = cdist(X_query, X_local)[0]
            k_eff = min(self.k_neighbors, len(dists))
            if k_eff < 1:
                k_eff = 1
                
            if k_eff >= len(dists):
                nn_idx = np.arange(len(dists))
            else:
                nn_idx = np.argpartition(dists, k_eff)[:k_eff]
                
            nn_dists = dists[nn_idx]
            nn_y = y_local[nn_idx]
            
            if self.weights == "distance":
                exact_mask = (nn_dists == 0)
                if np.any(exact_mask):
                    pred = nn_y[np.argmin(nn_dists)]
                else:
                    w = 1.0 / nn_dists
                    pred = np.sum(nn_y * w) / np.sum(w)
            else:
                pred = np.mean(nn_y)
            return float(pred)
            
        #批量分支預測
        if self.centers is not None and len(self.centers) > 0:
            try:
                clusters, _ = predict_cluster(X_query, self.centers)
            except Exception:
                clusters = np.zeros(n_query, dtype=int)
        else:
            clusters = np.zeros(n_query, dtype=int)
        
        preds = np.empty(n_query)
        
        unique_clusters = np.unique(clusters)
        for c in unique_clusters:
            idx = np.where(clusters == c)[0]
            X_q = X_query[idx]
            X_local = self.X_by_cluster.get(c, self.X_train)
            y_local = self.y_by_cluster.get(c, self.y_train)
            
            D = cdist(X_q, X_local)
            
            k_eff = min(self.k_neighbors, D.shape[1])
            if k_eff < 1:
                k_eff = 1
            
            if k_eff >= D.shape[1]:
                nn_idx = np.tile(np.arange(D.shape[1]), (len(X_q), 1))
            else:
                nn_idx = np.argpartition(D, k_eff, axis=1)[:, :k_eff]
            
            rows = np.arange(len(X_q))[:, None]
            nn_dists = D[rows, nn_idx]
            nn_y = y_local[nn_idx]
            
            if self.weights == "distance":
                exact = (nn_dists == 0)
                has_exact = exact.any(axis=1)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    w = np.where(nn_dists > 0, 1.0 / nn_dists, 0.0)
                w_sum = w.sum(axis=1, keepdims=True)
                w_sum = np.where(w_sum == 0, 1.0, w_sum)
                batch_pred = (w * nn_y).sum(axis=1) / w_sum.ravel()
                
                for qi in np.where(has_exact)[0]:
                    batch_pred[qi] = nn_y[qi, np.argmin(nn_dists[qi])]
            else:
                batch_pred = nn_y.mean(axis=1)
            
            preds[idx] = batch_pred
        
        return preds

    def predict_single(self, x):
        return float(self.predict(np.atleast_2d(x))[0])
