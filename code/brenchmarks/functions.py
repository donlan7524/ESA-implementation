import numpy as np

def sphere(x):
    x = np.asarray(x)
    return np.sum(x**2)

def ellipsoid(x):
    x = np.asarray(x)
    d = len(x)
    return np.sum(np.arange(1, d + 1) * (x**2))


def rosenbrock(x):
    x = np.asarray(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def ackley(x):
    x = np.asarray(x)
    d = len(x)

    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    t1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / d))
    t2 = -np.exp(sum2 / d)
    return t1 + t2 + 20 + np.e

def griewank(x):
    x = np.asarray(x)
    d = len(x)
    sumpart = np.sum(x**2) / 4000
    productpart = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
    return sumpart - productpart + 1

def shifted_sphere(x):
    x = np.asarray(x)
    shift = np.ones_like(x) * 3.0
    return np.sum((x - shift) ** 2)

try:
    from opfunu.cec_based import F102005, F162005, F192005
except ImportError:
    pass


class opfunu_wrapper:
    def __init__(self, cec_class, d):
        self.d = d
        self.func_instance = cec_class(ndim=d)
        self.lb = self.func_instance.lb
        self.ub = self.func_instance.ub

    def __call__(self, x):
        x = np.asarray(x)
        if len(x) != self.d:
            raise ValueError(f"Input dimension {len(x)} does not match expected dimension {self.d}.")
        return self.func_instance.evaluate(x)
    
def srr(d):
    return opfunu_wrapper(F102005, d)
def rhc1(d):
    return opfunu_wrapper(F162005, d)
def rhc2(d):
    return opfunu_wrapper(F192005, d)
