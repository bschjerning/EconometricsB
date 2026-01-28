import numpy as np
from numpy import random
from numpy import linalg as la
from tabulate import tabulate
from scipy.stats import norm
      
def q(beta,y,x):
    return - loglik_probit(beta,y,x)

def loglik_probit(beta, y, x):
    z = x@beta
    G = norm.cdf(z)

    # Make sure that no values are below 0 or above 1.
    h = np.sqrt(np.finfo(float).eps)
    G = np.clip(G, h, 1 - h)

    # Make sure g and y is 1-D array
    G = G.reshape(-1, )
    y = y.reshape(-1, )

    ll = y*np.log(G) + (1 - y)*np.log(1 - G)
    return ll

def starting_values(y,x):
    return la.solve(x.T @ x, x.T @ y )