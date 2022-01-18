import numpy as np
from numpy import random as random
from numpy import linalg as la


def forward_diff(func, x0, h=1.49e-08):
    # Take a forward step
    if x0 != 0:
        x1 = x0 * (1 + h)
    
    # If 0, then we need to take absolute step
    else:
        x1 = h

    step = x1 - x0
    grad = (func(x1) - func(x0))/step
    return grad
































# This is for vector difference, not in use.
def forward_diff_vec(func, x0, h=1.49e-08):
    f0 = func(x0)
    n = f0.size
    k = x0.size
    grad = np.zeros((n, k))
    for i in range(k):
        x1 = x0.copy()

        # Take a forward step
        if x0 != 0:
            x1[i] = x0[i] * (1 + h)
        
        # If 0, then we need to take absolute step
        else:
            x1[i] = h
        step = x1[i] - x0[i]
        grad[:, i] = (func(x1) - f0)/step
    return grad
