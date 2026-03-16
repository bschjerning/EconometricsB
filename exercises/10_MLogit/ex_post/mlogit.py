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
    v = util(theta, x)
    denom = np.sum(np.exp(v), axis=1, keepdims=False) # not keepdims => (N,) 1dim vector
    v_i = v[np.arange(N), y] # also a 1-dim vector 
    ll = v_i - np.log(denom)
    return ll


def util(theta, x):
    N, K = x.shape
    beta = theta.reshape(K, -1) # in case beta has been flattened e.g. by scipy.minimize 
    K, J_1 = beta.shape 
    J = J_1+1

    v_sub = x @ beta
    v = np.hstack([np.zeros((N,1)), v_sub])
    # full matrix of coefficients (including those for the normalized alternative )
    # beta = np.hstack([np.zeros((K,1)), theta]) # (K, J)

    # compute deterministic utility 
    # (matrix product of (N,K) and (K,J) matrices)
    # v = x @ beta # (N, J)

    # Substract maximum for numerical stability
    max_v = v.max(axis=1, keepdims=True) # keepdims: ensures (N,1) and not (N,) so that we can subtract (N,1) from (N,J) in the "natural" way
    assert v.ndim == max_v.ndim 
    v -= max_v

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
    
    v = util(theta, x)

    denom = np.sum(np.exp(v), axis=1, keepdims=True)
    
    # Conditional choice probabilites
    ccp = np.exp(v) / denom

    # log CCPs (can avoid the exponential in the numerator)
    logccp = v - np.log(denom)

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

    K, J_1 = theta.shape
    J = J_1 + 1

    xx = np.random.normal(size=(N,K-1)) 
    oo = np.ones((N,1)) # constant term 
    x  = np.hstack([oo,xx]) # full x matrix 

    z = np.zeros((K,1))
    beta = np.hstack([z, theta]) # full (K,J) matrix

    v = x @ beta 
    uni = np.random.uniform(size=(N,J))
    e = genextreme.ppf(uni, c=0)
    u = v + e # full utility

    y = u.argmax(axis=1) # observed, chosen alternative

    return y,x
