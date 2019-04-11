#损失函数
import numpy as np
#均方误差
def mse(y, t):
    return 0.5 * np.sum((y-t)**2)

#交叉熵
def cross_entropy_error(y, t):
    delta = 1e-7
    return  -np.sum(t * np.log(y + delta))