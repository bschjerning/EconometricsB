import numpy as np
from numpy import linalg as la
import pandas as pd 
from scipy import optimize
from tabulate import tabulate

def estimate(
        q: object, 
        theta0: list, 
        y: np.ndarray, 
        x: np.ndarray, 
        cov_type='Outer Product',
        options = {'disp': True},
        **kwargs
    ) -> dict:
    """Takes a function and returns the minimum, given start values and 
    variables to calculate the residuals.

    Args:
        q: The function to minimize. Must return an (N,) vector.
        theta0 (list): A list with starting values.
        y (np.array): Array of dependent variable.
        x (np.array): Array of independent variables.
        cov_type (str, optional): String for which type of variances to 
        calculate. Defaults to 'Outer Product'.
        options: dictionary with options for the optimizer (e.g. disp=True,
        which tells it to display information at termination.)

    Returns:
        dict: Returns a dictionary with results from the estimation.
    """
    
    # The minimizer can't handle 2-D arrays, so we flatten them to 1-D.
    theta0 = theta0.flatten()
    N = y.size

    # The objective function is the average of q(), 
    # but Q is only a function of one variable, theta, 
    # which is what minimize() will expect
    Q = lambda theta: np.mean(q(theta, y, x))

    # call optimizer
    result = optimize.minimize(
        Q, theta0, options=options,  **kwargs
        )
    
    cov, se = variance(q, y, x, result, cov_type)   

    # collect output in a dict 
    res = {
        'theta_hat': result.x,
        'se':       se,
        't_values': result.x / se,
        'cov':      cov,
        'success':  result.success, # bool, whether convergence was succesful 
        'nit':      result.nit, # no. algorithm iterations 
        'nfev':     result.nfev, # no. function evaluations 
        'fun':      result.fun # function value at termination 
    }
    return res

def variance(
        q, # function taking three inputs: q(theta, y, x) 
        y: np.ndarray, 
        x: np.ndarray, 
        result: dict, 
        cov_type: str
    ) -> tuple:
    """Calculates the variance for the likelihood function.

    Args:
        >> q: Function taking three inputs, q(theta,y,x), where we minimize the mean of q.
        >> y (np.ndarray): Dependent variable.
        >> x (np.ndarray): Independent variables.
        >> result (dict): Output from the function estimate().
        >> cov_type (str): Type of calculation to use in estimation.

    Returns:
        tuple: Returns the variance-covariance matrix and standard errors.
    """
    N = y.size
    N,K = x.shape
    thetahat = result.x
    P = thetahat.size

    # numerical gradients 
    f_q = lambda theta : q(theta,y,x) # as q(), but only a function of theta, whereas q also takes y,x as inputs
    s = centered_grad(f_q, thetahat)

    # "B" matrix 
    B = (s.T@s)/N
    
    # cov: P*P covariance matrix of theta 
    if cov_type == 'Hessian':
        A_inv = result.hess_inv
        cov = 1/N * A_inv
    elif cov_type == 'Outer Product':
        cov = 1/N * la.inv(B)
    elif cov_type == 'Sandwich':
        A_inv = result.hess_inv
        cov = 1/N * (A_inv @ B @ A_inv)

    # se: P-vector of std.errs. 
    se = np.sqrt(np.diag(cov))

    return cov, se


def centered_grad(f, x0: np.ndarray, h:float=1.49e-08) -> np.ndarray:
    '''centered_grad: numerical gradient calculator
    Args.
        f: function handle taking *one* input, f(x0). f can return a vector. 
        x0: P-vector, the point at which to compute the numerical gradient 

    Returns
        grad: N*P matrix of numericalgradients. 
    '''
    assert x0.ndim == 1, f'Assumes x0 is a flattened array'
    P = x0.size 

    # evaluate f at baseline 
    f0 = f(x0)
    N = f0.size

    # intialize output 
    grad = np.zeros((N, P))
    for i in range(P): 

        # initialize the step vectors 
        x1 = x0.copy()  # forward point
        x_1 = x0.copy() # backwards 

        # take the step for the i'th coordinate only 
        if x0[i] != 0: 
            x1[i] = x0[i]*(1.0 + h)  
            x_1[i] = x0[i]*(1.0 - h)
        else:
            # if x0[i] == 0, we cannot compute a relative step change, 
            # so we just take an absolute step 
            x1[i] = h
            x_1[i] = -h
        
        step = x1[i] - x_1[i] # the length of the step we took 
        grad[:, i] = ((f(x1) - f(x_1))/step).flatten()

    return grad


def print_table(
        theta_label: list,
        results: dict,
        headers:list = ["", "Beta", "Se", "t-values"],
        title:str = "Results",
        num_decimals:int = 4
    ) -> None:
    """Prints a nice looking table, must at least have coefficients, 
    standard errors and t-values. The number of coefficients must be the
    same length as the labels.

    Args:
        theta_label (list): List of labels for estimated parameters
        results (dict): The output from estimate()
        dictionary with at least the following keys:
            'theta_hat', 'se', 't_values'
        headers (list, optional): Column headers. Defaults to 
            ["", "Beta", "Se", "t-values"].
        title (str, optional): Table title. Defaults to "Results".
        num_decimals: (int) where to round off results (=None to disable)
    """
    assert len(theta_label) == len(results['theta_hat'])
    
    tab = pd.DataFrame({
       'theta': results['theta_hat'], 
        'se': results['se'], 
        't': results['t_values']
        }, index=theta_label)
    
    if num_decimals is not None: 
        tab = tab.round(num_decimals)
    
    # Print the table
    opt_outcome = 'succeeded' if results['success'] else 'failed'
    print(f'Optimizer {opt_outcome} after {results["nit"]} iter. ({results["nfev"]} func. evals.). Final criterion: {results["fun"]: 8.4g}.')
    print(title)
    return tab 