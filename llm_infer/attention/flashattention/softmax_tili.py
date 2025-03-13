import numpy as np


#1. python实现
def softmax_py(x):
    """
    x: 1-d numpy array
    """
    max_x = -np.inf #极小值
    #计算最大值
    for t in x:
        max_x = t if t > max_x else max_x
    
    #计算exp累加值
    sum_exp = 0
    for t in x:
        sum_exp += np.exp(t - max_x) #减去最大值是为了防止exp后溢出
    
    #计算softmax
    output = []
    for t in x:
        output.append(np.exp(t-max_x)/sum_exp)

    return output

#2. numpy实现
def softmax_np(x):
    """
    x: 1-d numpy array
    """
    max_x = np.max(x)
    sum_exp = np.sum(np.exp(x-max_x))
    output = np.exp(x-max_x)/sum_exp
    return output

#softmax with tiling
#在第一个循环中同时对最大值m以及softmax的分母d进行更新，从而减少了一个循环。
def softmax_tile(x):
    """
    x: 1-d numpy array
    """
    #计算最大值和累加值
    #max_x代表j-1的最大值，max_x_new代表j的最大值
    max_x = -np.inf
    sum_exp = 0
    for t in x:
        max_x_new = t if t > max_x else max_x
        sum_exp = sum_exp * np.exp(max_x - max_x_new) + np.exp(t - max_x_new)
        max_x = max_x_new #更新最大值

    #计算softmax
    output = []
    for t in x:
        output.append(np.exp(t-max_x)/sum_exp)

    return output