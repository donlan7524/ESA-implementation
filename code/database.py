import numpy as np
from scipy.stats import qmc

class Database:

    def __init__(self, d, bounds):
        self.d = d
        self.bounds = bounds
        self.X = np.empty((0, d))
        self.y = np.empty((0, 1))

    def lhs(self, n_samples, real_fitness_func):
        '''
        Args:
            n_samples: int, number of samples to generate
            real_fitness_func: callable, function to evaluate the fitness of the samples

        Returns:
            n_samples: int, number of samples generated

        '''
        sampler = qmc.LatinHypercube(d=self.d)
        sample = sampler.random(n=n_samples)
        scaled_sample = qmc.scale(sample, self.bounds[:, 0], self.bounds[:, 1])
        y_init = np.array([real_fitness_func(x) for x in scaled_sample])
        self.X = scaled_sample
        self.y = y_init
        
        return n_samples
    
    def add_sample(self, x, y):
        '''
        新增單一組已評估的數據至資料庫中

        Args:
            x: np.array, shape (d,)
            y: float, fitness value of the sample
        '''
        x_flat = np.asarray(x).ravel()
        self.X = np.vstack((self.X, x_flat))
        self.y = np.append(self.y, y)

    def getbest(self):
        '''
        獲取資料庫中當前的歷史最優解及其 fitness 值

        Returns:
            tuple, best sample and its fitness value
        '''
        best_index = np.argmin(self.y)
        return np.copy(self.X[best_index]), self.y[best_index]
    
    def get_nbest(self, n):
        '''
    獲取資料庫中前 n 個最優的樣本，按 fitness 由小到大排序

        Args:
            n: int, number of best samples to return   

        Returns:
            tuple, arrays of best samples and their fitness values
        '''    
        best_indices = np.argsort(self.y)[:n]
        return np.copy(self.X[best_indices]), np.copy(self.y[best_indices])
    
    def nearest_point(self, center, count):
        '''
        獲取距離指定中心點最近的 count 個歷史樣本

        Args:
            center: np.array, shape (d,)
            count: int, number of nearest points to return
        
        Returns:
            tuple, arrays of nearest samples and their fitness values
        '''
        count = min(count, len(self.y))
        distances = np.linalg.norm(self.X - center, axis=1)
        nearest_indices = np.argsort(distances)[:count]
        return np.copy(self.X[nearest_indices]), np.copy(self.y[nearest_indices])
    
    def get_samples_range(self, lb, ub):
        '''
        獲取落在指定邊界區間 [lb, ub] 內的所有歷史樣本

        Args:
            lb: np.array, lower bounds
            ub: np.array, upper bounds

        Returns:
            tuple, arrays of samples and their fitness values within the range
        '''
        mask = np.all((self.X >= lb) & (self.X <= ub), axis=1)
        return np.copy(self.X[mask]), np.copy(self.y[mask])
    
    def get_nearest_fitness(self, x):
        '''
        尋找與指定點 x 最接近之樣本的 fitness 值

        Args:
            x: np.array, shape (d,)

        Returns:
            float, the fitness of the nearest sample to x
        '''
        x_flat = np.asarray(x).ravel()
        dis = np.linalg.norm(self.X - x_flat, axis=1)
        nearest_index = np.argmin(dis)
        return self.y[nearest_index]
    
    def __len__(self):
        return len(self.y)
