import numpy as np
from scipy.optimize import minimize
from tabulate import tabulate
from numpy import linalg as la
import time

import numpy as np
from scipy.optimize import minimize, approx_fprime
from numpy import linalg as la
import time

def estimation(Qfun, theta0, deriv=1, cov_type='sandwich', parnames='', output=False): 
    """
    General M-estimation framework with automatic Hessian computation if not supplied.
    
    Parameters:
    - Qfun: Function returning Q (objective), dQ (gradient), and optionally Hessian
    - theta0: Initial parameter guess
    - deriv: 1 for gradients, 2 if Hessian is available; otherwise computed numerically
    - cov_type: Covariance matrix type ('sandwich', 'Ainv', 'Binv')
    
    Returns:
    - Dictionary containing estimates, standard errors, covariance matrix, and statistics
    """
    tic = time.perf_counter()

    # Define functions for optimization
    Q = lambda theta: Qfun(theta, out='Q')  
    dQ = lambda theta: Qfun(theta, out='dQ') if deriv > 0 else None
    hess = lambda theta: Qfun(theta, out='H') if deriv > 1 else None

    # Optimization
    if deriv > 1:
        res = minimize(fun=Q, jac=dQ, hess=hess, x0=theta0.flatten(), method='trust-ncg')
        res.hess_inv = la.inv(res.hess)
    else:
        res = minimize(fun=Q, jac=dQ, x0=theta0, method='bfgs')

        # Compute numerical Hessian directly inside the function
        epsilon = np.sqrt(np.finfo(float).eps)  # Default step size
        H = approx_fprime(res.x, dQ, epsilon)  # Apply finite differences on gradient
        H = 0.5 * (H + H.T)  # Ensure symmetry
        res.hess_inv = la.inv(H)

    theta_hat = np.array(res.x).reshape(-1, 1)
    s_i = Qfun(theta_hat, out='s_i')  # Individual score contributions
    cov = avar(s_i, res.hess_inv, cov_type)  
    se = np.sqrt(np.diag(cov)).reshape(-1, 1)  
    toc = time.perf_counter()

    # Collect results
    names = ['parnames', 'theta_hat', 'se', 't-values', 'cov', 'Q', 'time', 's_i']
    results = [parnames, theta_hat, se, theta_hat / se, cov, Q(theta_hat), toc - tic, s_i]
    res.update(dict(zip(names, results)))

    if output:
        table_keys = ['theta_hat', 'se', 't-values']
        if res['parnames']:
            table_keys.insert(0, 'parnames')

        print(tabulate({k: res[k] for k in table_keys}, headers="keys", floatfmt="10.5f"))
        print(f"\nObjective function: {res['Q']}")
        print(f"Iterations: {res.nit}, Function evals: {res.nfev}, Gradient evals: {res.njev}")
        print(f"Elapsed time: {res['time']:.4f} seconds")

    return res


def avar(s_i, Ainv, cov_type='sandwich'):
    """
    Computes the asymptotic variance-covariance matrix.
    
    Parameters:
    - s_i: (N, K) matrix of individual scores
    - Ainv: Inverse Hessian matrix
    - cov_type: Type of covariance estimation ('Ainv', 'Binv', 'sandwich')

    Returns:
    - Covariance matrix of parameter estimates.
    """
    n, K = s_i.shape
    B = (s_i.T @ s_i) / n  # Outer-product of scores

    if cov_type == 'Ainv':        
        return Ainv / n  
    elif cov_type == 'Binv':        
        return la.inv(B) / n  
    elif cov_type == 'sandwich':    
        return Ainv @ B @ Ainv / n  