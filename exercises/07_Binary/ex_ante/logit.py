import numpy as np
from numpy import random
from numpy import linalg as la
from scipy.stats import norm

name = 'Logit'

DOCHECKS = True 

def G(z): 
    # Fill in 
    return None

def q(theta, y, x): 
    # Fill in 
    return None

def loglikelihood(theta, y, x):

    # verify that inputs are as expected 
    if DOCHECKS: 
        assert np.isin(y, [0,1]).all(), f'y must be binary: found non-binary elements.'
        assert y.ndim == 1
        assert x.ndim == 2 
        N,K = x.shape 
        assert y.size == N
        assert theta.ndim == 1 
        assert theta.size == K 

    # Fill in 
    Gxb = None
    
    # 2. avoid log(0.0) errors
    Gxb = np.fmax(Gxb, 1e-8)     # truncate below at 0.00000001
    Gxb = np.fmin(Gxb, 1.0-1e-8) # truncate above at 0.99999999

    # Fill in 
    ll = None 

    return ll

def starting_values(y,x): 
    # Fill in 
    return None

def predict(theta, x): 
    # the "prediction" is the response probability, Pr(y=1|x)
    yhat = None # Fill in 
    return yhat 

def Ginv(u): 
    '''Inverse logistic cdf: takes inputs, u, in (0;1), and returns numbers in (-inf; inf)'''
    x = - np.log( (1.0-u) / u )
    return x

def sim_data(theta: np.ndarray, N:int) -> tuple: 
    '''sim_data: simulate a dataset of size N with true K-parameter theta

    Args. 
        theta: (K,) vector of true parameters (k=0 will always be a constant)
        N (int): number of observations to simulate 
    
    Returns
        tuple: y,x
            y (float): binary outcome taking values 0.0 and 1.0
            x: (N,K) matrix of explanatory variables
    '''

    # 0. unpack parameters from theta
    # (trivial, only beta parameters)
    beta = theta

    K = theta.size 
    assert K>1, f'Only implemented for K >= 2'
    
    # 1. simulate x variables, adding a constant 
    oo = np.ones((N,1))
    xx = np.random.normal(size=(N,K-1))
    x  = np.hstack([oo, xx]);
    
    # 2. simulate y values
    
    # 2.a draw error terms 
    uniforms = np.random.uniform(size=(N,))
    u = Ginv(uniforms)

    # 2.b compute latent index 
    ystar = x@beta + u
    
    # 2.b compute observed y (as a float)
    y = (ystar>=0).astype(float)

    # 3. return 
    return y, x
