import numpy as np


def sphere(x):
    return np.sum(np.asarray(x) ** 2)

def ellipsoid(x):
    x = np.asarray(x)
    d = len(x)
    if d == 1:
        return x[0]**2
    # 論文標準的高病態橢圓公式
    return np.sum((10 ** (6 * np.arange(d) / (d - 1))) * (x ** 2))

def simple_ellipsoid(x):
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
    return np.sum((np.asarray(x) - 3.0) ** 2)


srr_100_shift = None
srr_100_matrix = None
rhc1_100_matrix = None
rhc2_100_matrix = None

try:
    from opfunu.cec_based import F102005, F162005, F192005
except ImportError:
    pass

class opfunu_wrapper:
    def __init__(self, cec_class, d):
        global srr_100_shift, srr_100_matrix, rhc1_100_matrix, rhc2_100_matrix
        self.d = d
        
        if d == 100:
            if cec_class.__name__ == "F102005":
                if srr_100_shift is None:
                    state = np.random.RandomState(999)
                    H = state.randn(100, 100)
                    Q, _ = np.linalg.qr(H)
                    srr_100_matrix = Q
                    srr_100_shift = state.uniform(-5.0, 5.0, 100)
                self.func_instance = cec_class(ndim=100, f_shift=srr_100_shift, f_matrix=srr_100_matrix)
                self.func_instance.dim_supported = [10, 30, 50, 100]
            
            elif cec_class.__name__ == "F162005":
                if rhc1_100_matrix is None:
                    state = np.random.RandomState(888)
                    matrices = []
                    for _ in range(10):
                        H = state.randn(100, 100)
                        Q, _ = np.linalg.qr(H)
                        matrices.append(Q)
                    rhc1_100_matrix = np.vstack(matrices)
                self.func_instance = cec_class(ndim=100, f_matrix=rhc1_100_matrix)
                self.func_instance.dim_supported = [10, 30, 50, 100]
                
            elif cec_class.__name__ == "F192005":
                if rhc2_100_matrix is None:
                    state = np.random.RandomState(777)
                    matrices = []
                    for _ in range(10):
                        H = state.randn(100, 100)
                        Q, _ = np.linalg.qr(H)
                        matrices.append(Q)
                    rhc2_100_matrix = np.vstack(matrices)
                self.func_instance = cec_class(ndim=100, f_matrix=rhc2_100_matrix)
                self.func_instance.dim_supported = [10, 30, 50, 100]
            else:
                self.func_instance = cec_class(ndim=d)
        else:
            self.func_instance = cec_class(ndim=d)
            
        self.lb = self.func_instance.lb
        self.ub = self.func_instance.ub

    def __call__(self, x):
        x = np.asarray(x)
        if len(x) != self.d:
            raise ValueError(f"Input dimension {len(x)} does not match expected dimension {self.d}.")
        return self.func_instance.evaluate(x)
    
def srr(d):   return opfunu_wrapper(F102005, d)
def rhc1(d):  return opfunu_wrapper(F162005, d)
def rhc2(d):  return opfunu_wrapper(F192005, d)