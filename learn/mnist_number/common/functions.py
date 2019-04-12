#损失函数
import numpy as np
#均方误差
def mse(y, t):
    return 0.5 * np.sum((y-t)**2)

#交叉熵
def cross_entropy_error(y, t):
    delta = 1e-7
    return  -np.sum(t * np.log(y + delta))

#sigmod 激活函数
def sigmod(x):
    return 1/(np.exp(-x) + 1)

#softmax 输出函数，一般用于分类
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))