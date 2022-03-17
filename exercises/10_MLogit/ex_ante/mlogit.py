import numpy as np
from scipy.stats import genextreme # generalized extreme value distribution 

def q(theta, y,  x): 
    '''Criterion function: negative loglikelihood'''
    return - loglike(theta, y, x)


def loglike(theta, y, x):
    """Inputs data and coefficients, and outputs a vector with
    log choice probabilities dependent on actual choice from y vector.

    Args:
        theta (np.array): Coefficients, or weights for the x array.
        y (np.array): Dependent variable
        x (np.array): Independent variable

    Returns:
        np.array: Log choice probabilities with dimensions (n, 1)
    """
    N, K = x.shape

    # FILL IN 
    ll = None # loglikelihood: (N,) vector
    return ll


def util(theta, x):
    N, K = x.shape
    beta = theta.reshape(K, -1) # sometimes, theta may be flattened, e.g. by scipy.minimize: so we make it a matrix again 
    K, J_1 = beta.shape # the second dimension must be J-1 
    J = J_1+1


    # FILL IN 
    # 1. compute v (observable utilities, matrix product)
    # 2. add column of zeros (for normalized alternative)
    v = None 

    # Substract maximum for numerical stability
    max_v = v.max(axis=1, keepdims=True) # keepdims: ensures (N,1) and not (N,) so that we can subtract (N,1) from (N,J) in the "natural" way
    v = v - max_v # since max_v is (N,1) and v is (N,J), the subtraction works out 

    return v


def choice_prob(theta, x):
    """Takes the coefficients and covariates and outputs choice probabilites 
    and log choice probabilites. The utilities are max rescaled before
    choice probabilities are calculated.

    Args:
        theta (np.array): Coefficients, or weights for the x array
        x (np.array): Dependent variables.

    Returns:
        (tuple): Returns choice probabilities and log choice probabilities,
            both are np.array. 
    """
    assert x.ndim == 2, f'x must be 2-dimensional'
    
    # FILL IN 
    v = None # compute utility (fill out the util() function and use it)
    denom = None # compute the denominator of the choice probability (make sure it is (N,1) and not (N,))
    
    # Conditional choice probabilites
    ccp = None # exp(v) / [sum exp(v)]

    # log CCPs 
    logsumexpv = None # log[sum(exp(v))]: make sure that it is (N,1) and not (N,)
    logccp = v - logsumexpv # subtracting an (N,1) from an (N,J) matrix! 

    return ccp, logccp

def starting_values(y, x, J:int): 
    N,K = x.shape
    theta0 = np.zeros(K,J-1)
    return theta0

def sim_data(theta, N: int):
    """Takes input values n and j to specify the shape of the output data. The
    K dimension is inferred from the length of theta. Creates a y column vector
    that are the choice that maximises utility, and a x matrix that are the 
    covariates, drawn from a random normal distribution.

    Args:
        N (int): Number of households.
        J (int): Number of choices.
        theta (np.array): True parameters (K, J-1)

    Returns:
        tuple: y,x 
    """

    assert theta.ndim == 2, 'theta must be 2-dimensional'
    K, J_1 = theta.shape
    J = J_1 + 1

    xx = None # draw (N,K-1) matrix of random normal covariates
    oo = np.ones((N,1)) # constant term 
    x  = np.hstack([oo,xx]) # full x matrix 

    # FILL IN 
    beta = None # should be a (K,J) matrix: first column = zeros, remainder = theta 
    v = None # observable utility, (N,J): use a matrix product
    e = None # use genextreme.ppf(uni, c=0) on an (N,J) matrix of random uniform draws
    u = None # full utility 

    # observed, chosen alternative
    y = None # take the argmax row-wise (i.e. over j=0,...,J-1): verify that y is (N,) and not (N,1)

    return y,x
