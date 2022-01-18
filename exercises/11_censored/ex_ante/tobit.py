import numpy as np 
from scipy.stats import norm

name = 'Tobit'

def q(theta, y, x): 
    return None # Fill in 

def loglikelihood(theta, y, x): 
    assert True # FILL IN: add some assertions to make sure that dimensions are as you assume 

    # unpack parameters 
    b = theta[:-1] # first K parameters are betas, the last is sigma 
    sig = np.abs(theta[-1]) # take abs() to ensure positivity (in case the optimizer decides to try it)

    ll = None # fill in 

    return ll

def starting_values(y,x): 
    '''starting_values
    Returns
        theta: K+1 array, where theta[:K] are betas, and theta[-1] is sigma (not squared)
    '''
    b_ols = None # OLS estimates as starting values 
    sigmahat = None # OLS estimate of sigma = sqrt(sigma^2) as starting value 
    theta0 = np.append(b_ols, sigmahat)
    return theta0 

def predict(theta, x): 
    '''predict(): the expected value of y given x 
    Returns E, E_pos
        E: E(y|x)
        E_pos: E(y|x, y>0) 
    '''
    # Fill in 
    E = None
    Epos = None
    return E, Epos

def sim_data(theta, N:int): 
    b = theta[:-1]
    sig = theta[-1]

    # FILL IN 
    x = None 
    y = None 

    return y,x
