import numpy as np
import graphlab


def getRss(predict, actual):
    r = sum(np.square(np.subtract(predict, actual)))
    return r

