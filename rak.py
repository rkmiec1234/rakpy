import numpy as np


def getRss(predict, actual):
    r = sum(np.square(np.subtract(predict, actual)))
    return r

