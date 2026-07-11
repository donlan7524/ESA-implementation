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

class CustomRotatedFunction:
    def __init__(self, d, func_type, seed):
        self.d = d
        self.func_type = func_type
        
        # 綁定一個固定的隨機種子，確保每次產生一樣的平移與旋轉矩陣
        state = np.random.RandomState(seed)
        
        # 產生平移向量 (Shift)
        self.shift = state.uniform(-5.0, 5.0, d)
        
        # 產生正交旋轉矩陣 (Rotation Matrix)
        H = state.randn(d, d)
        Q, _ = np.linalg.qr(H)
        self.matrix = Q

    def __call__(self, x):
        x = np.asarray(x)
        if len(x) != self.d:
            raise ValueError(f"輸入維度 {len(x)} 與期望維度 {self.d} 不符。")
            
        # 1. 平移 (Shift)
        z = x - self.shift
        # 2. 旋轉 (Rotate)
        z = np.dot(z, self.matrix)
        
        # 3. 帶入核心函數計算
        if self.func_type == 'SRR':
            # Shifted Rotated Rastrigin
            return np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)
            
        elif self.func_type == 'RHC':
            # Rotated High Conditioned Elliptic
            if self.d == 1:
                return z[0]**2
            weights = 10 ** (6 * np.arange(self.d) / (self.d - 1))
            return np.sum(weights * (z ** 2))
            
        else:
            raise ValueError(f"未知的函數類型: {self.func_type}")

# 建立實例的封裝函數
# 為了對應你的 main.py 邏輯，這裡固定給定 seed 確保同一個 function 矩陣不變
def srr(d):   return CustomRotatedFunction(d, 'SRR', seed=999)

# 備註：你在原本程式中 RHC1, RHC2 呼叫的是 F16, F19
# 這裡我幫你實作了標準的「旋轉高病態橢圓 (Rotated High Conditioned Ellipsoid)」
# 為了讓你原本的 RHC1, RHC2 都有東西跑，我用不同的 seed 區分它們
def rhc1(d):  return CustomRotatedFunction(d, 'RHC', seed=888)
def rhc2(d):  return CustomRotatedFunction(d, 'RHC', seed=777)