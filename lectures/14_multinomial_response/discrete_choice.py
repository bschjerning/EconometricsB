import numpy as np
import pandas as pd
import mestim as M
from tabulate import tabulate
from numpy import linalg as la
from numpy import random
from scipy.stats import genextreme

import numpy as np
from scipy.stats import genextreme

def sim_data(N, J, theta):
    """
    Simulate data for a Conditional Logit model.

    Parameters:
    - N: Number of individuals
    - J: Number of alternatives
    - theta: True parameter values (K, 1)

    Returns:
    - Dictionary with choices (y) and covariates (x)
    """

    k = theta.size  # Number of covariates
    
    # Generate covariates with increasing means across alternatives
    x = np.random.normal(size=(N, J, k)) + np.linspace(3, 5, J).reshape(1, J, 1)
    
    # Compute deterministic utility
    v = utility(theta, x)
    
    # Add extreme value Type I errors (logit-distributed shocks)
    e = genextreme.ppf(np.random.uniform(size=(N, J)), c=0)
    u = v + e  # Total utility
    
    # Choose alternative that maximizes utility
    y = u.argmax(axis=1)
    
    return {'y': y, 'x': x}

def clogit(y, x, cov_type='Ainv', theta0=None, deriv=1, quiet=False,  xvars=None): 
    """
    Estimates a Conditional Logit model using M-estimation.
    
    Parameters:
    - y: (N,) vector of observed choices (0, 1, ..., J-1)
    - x: (N, J, K) covariate matrix (varying across alternatives)
    - cov_type: Type of variance estimation ('Ainv', 'Binv', 'sandwich')
    - theta0: Initial parameter guess (default: zeros)
    - deriv: Order of derivative (0 for none, 1 for first-order)
    - quiet: Suppress output if True
    
    Returns:
    - res: Dictionary with estimates, standard errors, and diagnostics
    """
    N, J, K, _, _, xvars = labels(x, xvars)  # Extract dimensions and variable names
    
    Qfun = lambda theta, out: Q_clogit(theta, y, x, out)  # Define sample objective function
    theta0 = np.zeros(K) if theta0 is None else theta0  # Default start values
    res = M.estimation(Qfun, theta0, deriv, cov_type, parnames=xvars)  # Estimate via MLE
    res.update({'yvar': 'y', 'xvars': xvars, 'N': N, 'K': K, 'J': J})  # Store model details

    if not quiet:    
        print('Conditional Logit Estimation')
        print('Initial log-likelihood:', -Qfun(theta0, 'Q'))
        print('Initial gradient:\n', -Qfun(theta0, 'dQ'))
        print_output(res)

    return res

def Q_clogit(theta, y, x, out='Q'):
    """
    Sample log-likelihood and its derivatives for Conditional Logit.
    
    Parameters:
    - theta: (K, 1) parameter vector
    - y: (N,) observed choices
    - x: (N, J, K) covariates (varying across alternatives)
    - out: 'Q' for negative log-likelihood, 's_i' for individual scores, 'dQ' for gradient of Q.

    Returns:
    - Negative mean log-likelihood (if out='Q')
    - Individual score contributions (if out='s_i')
    - Gradient (if out='dQ')
    """
    
    N, J, K = x.shape
    v = utility(theta, x)  # Compute deterministic utility v_ij = x_ij * beta
    ll_i = logccp(v, y)   # Compute log-likelihood contribution

    if out == 'Q': 
        return -np.mean(ll_i)  # Sample objective function (negative mean log-likelihood)

    # Compute (s_i) or gradient (dQ)
    p = ccp(v)  # Compute choice probabilities p_ij
    x_iy = x.reshape(N * J, K)[y + J * np.arange(N), :]  # Extract covariates for chosen alternative
    s_i = x_iy - np.sum(p.reshape(N, J, 1) * x, axis=1)  # Score function: residual of x_iy - E[x|choice]

    return s_i if out == 's_i' else -np.mean(s_i, axis=0)  # Return individual scores or gradient (dQ)

def utility(theta, x):
    N, J, K = x.shape       # N: Individuals, J: Alternatives, K: Covariates
    u = x @ theta           # Compute deterministic utility v_ij = x_ij * beta
    return u.reshape(N, J)  # Reshape to match (N, J) structure

def logsum(v, sigma=1): 
	# Expected max over iid extreme value shocks with scale parameter sigma
	# Logsum is reentered around maximum to obtain numerical stability (avoids overflow, but accepts underflow)
	max_v = v.max(axis=1).reshape(-1, 1)
	return max_v + sigma*np.log(np.sum(np.exp((v-max_v)/sigma), 1)).reshape(-1, 1)
	
def logccp(v, y=None, sigma=1):
    # Log of conditional choice probabilities 
    # If y=None return logccp corresponding to all choices
    # if y is Nx1 vector of choice indexes, return likelihood

    ev=logsum(v, sigma) 	# Expected utility (always larger than V)
    if y is not None:  		
    	N, J=v.shape
    	idx=y[:,] + J*np.arange(0, N)
    	v=v.reshape(N*J, 1)[idx] 	# pick choice specific values corresponding to y 
    return v/sigma - ev

def ccp(v, y=None, sigma=1):
    # Conditional choice probabilities
    return np.exp(logccp(v, y, sigma))

def labels(x, xvars=None, alternatives=None):
    """
    Extract dimensions and generate labels for alternatives and variables.

    Parameters:
    - x: (N, J, K) covariate matrix
    - xvars: List of variable names (default: 'var0', 'var1', ...)
    - alternatives: List of alternative names (default: 'alt0', 'alt1', ...)

    Returns:
    - N: Number of individuals
    - J: Number of alternatives
    - K: Number of covariates
    - palt: List of probability labels ('p0', 'p1', ...)
    - xalt: List of alternative labels ('alt0', 'alt1', ...)
    - xvars: List of covariate names ('var0', 'var1', ...)
    """

    N, J, K = x.shape

    if xvars is None:
        xvars = ['var' + str(i) for i in range(K)]

    if alternatives is None:
        palt = ['p' + str(i) for i in range(J)]
        xalt = ['alt' + str(i) for i in range(J)]
    else:
        palt = ["p_" + str(alt) for alt in alternatives]
        xalt = ["x_" + str(alt) for alt in alternatives]

    return N, J, K, palt, xalt, xvars

def APE_var(theta, x, m=0, xvars=None, alternatives=None, quiet=False):
    """
    Compute Average Partial Effects (APE) for a change in attribute m.

    Parameters:
    - theta: (K, 1) estimated coefficients
    - x: (N, J, K) attribute matrix (varying across alternatives)
    - m: index of the attribute for which APE is computed
    - quiet: if False, prints the APE results

    Returns:
    - E: (J, J) matrix of average partial effects where:
        - Rows (k) correspond to the alternative whose attribute x_{ik} is changed.
        - Columns (j) correspond to the alternative whose probability p_{ij} is affected.
        - Each entry E[k, j] represents the effect of a change in x_{ik} on the probability p_{ij}.
    """
    
    N, J, K, palt, xalt, xvars = labels(x, xvars, alternatives) # Extract dimensions and labels
    
    # Compute APE: Expected marginal effects of x_{ik} on p_{ij}
    p = ccp(utility(theta, x))  # Compute choice probabilities p_ij
    E = np.empty((J, J)) # Initialize APE matrix     
    for j in range(J):  # Loop over alternatives (j)
        for k in range(J):  # Loop over alternatives (k)
            E[k, j] = np.mean(p[:, j] * theta[m] * (1 * (j == k) - p[:, k]), axis=0)

    # Print APE results if quiet=False
    if not quiet:  
        print(f"\nAPE: Average change in {palt}\n    - w.r.t. change in {xalt} \n    - for attribute m={m} ({xvars[m]}) with coefficient θ_{m}={theta[m].round(4)}")
        print(tabulate(np.c_[xalt, E], headers=palt, floatfmt="10.5f"))

    return E

def Ematrix_var(theta, x, m=0, xvars=None, alternatives=None, quiet=False):
	# matrix of elasticities with respect ot a change in attribute m
	N, J, K, palt, xalt, xvars = labels(x, xvars, alternatives)
	p=ccp(utility(theta, x))
	E=np.empty((J,J))
	for j in range(J):
	    for k in range(J):
	        E[k, j]=np.mean(x[:,k,m]*theta[m]*(1*(j==k)-p[:,k]), axis=0)
	if not quiet: 
	    print('\nElasticity wrt change in', xvars[m])
	    print(tabulate(np.c_[xalt, E], headers=palt,floatfmt="10.5f"))
	return E

def Ematrix_own(theta, x, xvars=None, alternatives=None, quiet=False):
    # Own elasticity: % change in prob of alternative j wrt % change in attribute of same alternative j
    # done for for each variable in x  
    N, J, K, palt, xalt, xvars = labels(x, xvars, alternatives)
    p=ccp(utility(theta, x))
    E_own=np.empty((J, K))
    for iJ in range(J):
        for iK in range(K):
            E_own[iJ, iK]=np.mean(x[:,iJ,iK]*theta[iK]*(1-p[:,iJ]), axis=0)
    if not quiet: 
        print('\nOwn elasticity')
        print(tabulate(np.c_[xalt, E_own], headers=xvars,floatfmt="10.5f"))
    return E_own

def Ematrix_cross(theta, x, xvars=None, alternatives=None, quiet=False):
    # Cross elasticity:  % change in prob of alternative j wrt % change in attribute of other alternative k ne j
    # done for each variable in x  
    N, J, K, palt, xalt, xvars = labels(x, xvars, alternatives)
    p=ccp(utility(theta, x))
    E_cross=np.empty((J, K))
    for iJ in range(J):
        for iK in range(K):
            E_cross[iJ, iK]=np.mean(x[:,iJ,iK]*theta[iK]*(-p[:,iJ]), axis=0)
    if not quiet: 
        print('\nCross-Elasticity')
        print(tabulate(np.c_[xalt, E_cross], headers=xvars,floatfmt="10.5f"))

def print_output(res, cols=['parnames','theta_hat', 'se', 't-values', 'jac']): 
    print('Dep. var. :', res['yvar'], '\n') 

    # Print columns that exist in res
    valid_cols = [k for k in cols if k in res]
    table = {k: res[k] for k in valid_cols}    
    print(tabulate(table, headers="keys", floatfmt="10.5f"))

    print('# of observations :', res['N'])
    print('# log-likelihood. :', - res['Q']*res['N'], '\n')
    print ('Iteration info: %d iterations, %d evaluations of objective, and %d evaluations of gradients' 
        % (res.nit,res.nfev, res.njev))
    print(f"Elapsed time: {res['time']:0.4f} seconds")
    print('')
