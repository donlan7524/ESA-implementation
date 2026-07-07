import numpy as np
from collections import deque

class UCB1Selector:
    '''
    score = 平均報酬 + c * sqrt(2 ln t / n_a)
    透過估算回報的上限，給予較少被使用的算子更高的探索機會
    '''
    def __init__(self, n_actions, c=0.5, rng=None):
        self.n_actions = n_actions
        self.rng = rng if rng is not None else np.random.default_rng()
        self.c = c
        self.t = 0
        self.n = np.zeros(n_actions)
        self.mean = np.zeros(n_actions)
 
    def select(self, state, feasible=None):
        feas = (np.ones(self.n_actions, bool) if feasible is None
                else np.asarray(feasible, bool))
        untried = feas & (self.n == 0)         # 每個可行動作先各試一次
        if untried.any():
            return int(self.rng.choice(np.flatnonzero(untried)))
        ucb = self.mean + self.c * np.sqrt(2.0 * np.log(max(self.t, 1)) / self.n)
        return int(np.argmax(self._mask(ucb, feas)))
 
    def update(self, state, action, reward, next_state):
        self.t += 1
        self.n[action] += 1
        self.mean[action] += (reward - self.mean[action]) / self.n[action]
 
 
