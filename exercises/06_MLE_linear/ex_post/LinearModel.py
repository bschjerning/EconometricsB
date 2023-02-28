# OLS module 
import numpy as np 
import numpy.linalg as la 

def q(theta: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray: 
    '''q: criterion function, the negative log likelihood
    '''
    return -loglikelihood(theta, y, x)

def loglikelihood(theta: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.array:
    """The likelihood criterion function, returns an array with the
    values from the likelihood criterion.

    Args:
        theta (np.ndarray): A list that contains the beta values and the sigma2
        y (np.array): Depentent variable
        x (np.array): Independent variables

    Returns:
        [np.array]: Array of likelihood values from the likelihood criterion.
    """
    # Unpack values
    N,K = x.shape
    assert len(theta) == K+1, f'Theta should have K+1={K+1} values (K regressors + 1 for sigma2)'

    beta = theta[:-1] # first K values 
    sigma2 = theta[-1]*theta[-1] # last element
    
    # Make sure inputs has correct dimensions
    # (the optimizer sometimes flattens parameters during estimation)
    beta = beta.reshape(-1, 1) 
    y = y.reshape(-1, 1)

    residual = y - x @ beta
    ll = -0.5 * np.log(sigma2) - 0.5 * residual*residual  / sigma2
    return ll

def starting_values(y, x):
    '''starting_values: 
    Args.
        y: N-vector (outcome)
        x: (N,K) matrix (regressors) 
    Returns
        theta: (K+1)-vector of OLS estimates, 
    '''
    # Make sure that y and x are 2-D.
    assert x.ndim == 2, f'x must be 2-dimensional'
    assert x.shape[0] == y.size, f'x and y must have same number of rows'
    N = y.size 
    N_,K = x.shape 

    # ensure y is not flattened
    y = y.reshape(-1, 1)

    # Estimate beta
    b_hat = la.inv((x.T@x))@(x.T@y)

    # Calculate standard errors
    # (non-robust version)
    residual = y - x@b_hat
    sigma2 = residual.T@residual/(N - K)
    sigma = np.sqrt(sigma2)

    # we do not require standard errors, but the formulas are nice enough :) 
    #cov = sigma2*la.inv(x.T@x)
    #se = np.sqrt(cov.diagonal()).reshape(-1, 1)  # The diagonal method returns 1d array.

    # Return osl estimates in a single array, 
    starting_vals = np.vstack([b_hat, sigma])
    
    return starting_vals


def sim_data(N: int, theta, rng):    
    assert theta.ndim == 2 # FIXME: Switch to 1-dim theta + change err to be (N,) rather than (N,1)
    
    # Unpack parameters
    K = len(theta) - 1
    beta = theta[:K]
    sigma = theta[-1]
    
    # Simalute x-values as N(0,1) and add a constant term 
    const = np.ones((N, 1))
    x0 = rng.normal(size=(N, K - 1))  # We substract 1, as the first is constant
    x = np.hstack((const, x0))
    
    # Simulate error term 
    err = rng.normal(scale=sigma, size=(N, 1))
    
    # Simulate y-values
    y = x @ beta + err
    return y, x