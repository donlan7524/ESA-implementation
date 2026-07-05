import numpy as np
from scipy.stats import qmc

class Database:

    def __init__(self, d, bounds):
        self.d = d
        self.bounds = bounds
        self.X = np.empty((0, d))
        self.y = np.empty((0, 1))

    def lhs(self, n_samples, real_fitness_func):
        sampler = qmc.LatinHypercube(d=self.d)
        sample = sampler.random(n=n_samples)
        scaled_sample = qmc.scale(sample, self.bounds[:, 0], self.bounds[:, 1])
        y_init = np.array([real_fitness_func(x) for x in scaled_sample])
        self.X = scaled_sample
        self.y = y_init
        
        return n_samples
    
    def add_sample(self, x, y):

        x_flat = np.asarray(x).ravel()
        self.X = np.vstack((self.X, x_flat))
        self.y = np.append(self.y, y)

    def getbest(self):
        best_index = np.argmin(self.y)
        return np.copy(self.X[best_index]), self.y[best_index]
    
    def get_nbest(self, n):
        best_indices = np.argsort(self.y)[:n]
        return np.copy(self.X[best_indices]), np.copy(self.y[best_indices])
    
    def nearest_point(self, center, count):
        '''
        Args:
            center: np.array, shape (d,)
            count: int, number of nearest points to return
        
        '''
        count = min(count, len(self.y))
        distances = np.linalg.norm(self.X - center, axis=1)
        nearest_indices = np.argsort(distances)[:count]
        return np.copy(self.X[nearest_indices]), np.copy(self.y[nearest_indices])
    
    def get_samples_range(self, lb, ub):
        mask = np.all((self.X >= lb) & (self.X <= ub), axis=1)
        return np.copy(self.X[mask]), np.copy(self.y[mask])
    
    def get_fitness(self, x):

        x_flat = np.asarray(x).ravel()
        dis = np.linalg.norm(self.X - x_flat, axis=1)
        nearest_index = np.argmin(dis)
        return self.y[nearest_index]
    
    def __len__(self):
        return len(self.y)
