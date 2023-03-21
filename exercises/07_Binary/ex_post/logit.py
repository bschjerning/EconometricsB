import numpy as np
from numpy import random
from numpy import linalg as la
from scipy import optimize
from scipy.stats import norm
from tabulate import tabulate

name = 'Logit'

DOCHECKS = True 

def G(z): 
    Gz = 1. / (1. + np.exp(-z))
    return Gz

def q(theta, y, x): 
    return -loglikelihood(theta, y, x)

def loglikelihood(theta, y, x):

    if DOCHECKS: 
        assert np.isin(y, [0,1]).all(), f'y must be binary: found non-binary elements.'
        assert y.ndim == 1
        assert x.ndim == 2 
        N,K = x.shape 
        assert y.size == N
        assert theta.ndim == 1 
        assert theta.size == K 

    # 0. unpack parameters 
    # (trivial, we are just estimating the coefficients on x)
    beta = theta 
    
    # 1. latent index
    z = x@beta
    Gxb = G(z)
    
    # 2. avoid log(0.0) errors
    h = 1e-8 # a tiny number 
    Gxb = np.fmax(Gxb, h)     # truncate below at 1e-8 
    Gxb = np.fmin(Gxb, 1.0-h) # truncate above at 0.99999999

    ll = (y==1)*np.log(Gxb) + (y==0)*np.log(1.0 - Gxb) 
    return ll

def Ginv(u): 
    '''Inverse logistic cdf: u should be in (0;1)'''
    x = - np.log( (1.0-u) / u )
    return x

def starting_values(y,x): 
    b_ols = la.solve(x.T@x, x.T@y)
    return b_ols*4.0

def predict(theta, x): 
    # the "prediction" is the response probability, Pr(y=1|x)
    yhat = G(x@theta) 
    return yhat 

def sim_data(theta: np.ndarray, N:int): 
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
