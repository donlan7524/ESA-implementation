# -*- coding: utf-8 -*-
import os

# "simple": 論文原版 RBF
# "hpo": 優化版 HPO RBF
RBF_MODE = os.environ.get("ESA_RBF_MODE", "hpo").lower()

if RBF_MODE == "hpo":
    from RBF_hpo import RBF as RBF
    from RBF_hpo import distance as distance
elif RBF_MODE == "clustered_knn":
    from KNN_surrogate import ClusteredKNN as RBF
    from RBF_simple import distance as distance
elif RBF_MODE == "mlp":
    from MLP_hpo import MLP_HPO as RBF
    from RBF_simple import distance as distance
elif RBF_MODE == "gp":
    from GP_surrogate import GP as RBF
    from RBF_simple import distance as distance
else:
    from RBF_simple import RBF as RBF
    from RBF_simple import distance as distance