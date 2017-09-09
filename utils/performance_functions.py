import numpy as np


def rmsle(predicted, real):
    result_sum = 0.0
    for x in range(len(predicted)):
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        result_sum = result_sum + (p - r)**2
    return (result_sum/len(predicted))**0.5

