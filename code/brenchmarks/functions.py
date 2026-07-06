import numpy as np

def sphere(x):
    x = np.asarray(x)
    return np.sum(x**2)

def shifted_sphere(x):
    x = np.asarray(x)
    shift = np.ones_like(x) * 3.0
    return np.sum((x - shift) ** 2)
