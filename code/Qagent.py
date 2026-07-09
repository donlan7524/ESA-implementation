import numpy as np

class Qlearning:

    def __init__(self, alpha=0.1, gamma=0.9, T=1.0, rng=None):
        '''
        Args:
        
        
        '''
        self.alpha = alpha
        self.gamma = gamma
        self.T = T
        self.rng = rng if rng is not None else np.random.default_rng()

        self.qtable = np.full((10, 5), 0.25)

    def get_initial_qtable(self):
        return self.rng.choice([0,2,4,6,8])

    def select_action(self, state):
        '''
        Args:
            state (int): current state index (0 to 7)
        Returns:
            action (int): selected action index (0 to 3)
        '''
        q_values = self.qtable[state]
        # Subtract max for numerical stability to prevent overflow
        exp_q = np.exp((q_values - np.max(q_values)) / self.T)
        probs = exp_q / np.sum(exp_q)
        return self.rng.choice(5, p=probs)
    
    def q_update(self, state, action, reward, next_state):
        '''
        Args:
            state (int): previous state index (0 to 7)
            action (int): action taken (0 to 3)
            reward (float): immediate reward obtained (1.0 for success, 0.0 for failure)
            next_state (int): next state index (0 to 7)
        '''
        max_next_q = np.max(self.qtable[next_state])
        td_target = reward + self.gamma * max_next_q
        self.qtable[state, action] += self.alpha * (td_target - self.qtable[state, action])

    def next_state(self, action, success):
        return action*2 + 1 if success else action*2
    
    def update_T(self, T):
        self.T = T
        
    def compute_reward(self,is_success,n_evals):
        '''
        以FE為單位的reward：
        成功 -> 1 / 消耗的真實評估次數，失敗 -> 0。
        TRLS(3 FE) 成功一次只拿 1/3，
        和其他 1 FE 策略在相同預算下的期望報酬比較。
        '''
        if not is_success or n_evals <= 0:
            return 0.0
        return 1.0/n_evals