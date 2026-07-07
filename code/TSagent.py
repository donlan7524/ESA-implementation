import numpy as np

class DiscountedTS:
    """
    折扣型 Thompson sampling：
    每個動作維護 Beta(1 + a_cnt, 1 + b_cnt)，每次更新前所有計數乘上 decay，
    越舊的觀測權重越小 -> 同時處理探索/利用與非平穩性(近期解的影響力較大)。
    reward 需落在 [0,1]
    """
 
    def __init__(self, n_actions, decay=0.98, rng=None):
        self.n_actions = n_actions
        self.rng = rng if rng is not None else np.random.default_rng()
        self.decay = decay
        self.a_cnt = np.zeros(n_actions)
        self.b_cnt = np.zeros(n_actions)
 
    def select(self, state, feasible=None):
        samples = self.rng.beta(1.0 + self.a_cnt, 1.0 + self.b_cnt)
        return int(np.argmax(self._mask(samples, feasible)))
 
    def update(self, state, action, reward, next_state):
        r = float(np.clip(reward, 0.0, 1.0))
        self.a_cnt *= self.decay
        self.b_cnt *= self.decay
        self.a_cnt[action] += r
        self.b_cnt[action] += 1.0 - r
 