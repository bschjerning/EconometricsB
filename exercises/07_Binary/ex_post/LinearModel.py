import numpy as np
from numpy import linalg as la
import pandas as pd 


def estimate( 
        y, x, transform:str='', T=None, robust_se=False
    ) -> list:
    """Uses the OLS to perform a regression of y on x, and compute std.errs.

    Args:
        y (np.array): Dependent variable (Needs to have shape 2D shape)
        x (np.array): Independent variable (Needs to have shape 2D shape)
        z (None or np.array): Instrument array (Needs to have same shape as x)
        >> transform (str, optional): Defaults to ''. If the data is 
        transformed in any way, the following transformations are allowed:
            '': No transformations (default)
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation.
        >> T (int, optional): If panel data, T is the number of time periods in
        the panel, and is used for estimating the variance. Defaults to None.
        >> robust_se (bool): Calculates robust standard errors if True.
        Defaults to False.

    Returns:
        list: Returns a dictionary with the following variables:
        'b_hat', 'se', 'sigma2', 't', 'R2', 'cov'
    """
    
    assert y.ndim == 1, 'Input y must be 2-dimensional'
    assert x.ndim == 2, 'Input x must be 2-dimensional'
    N,K = x.shape
    assert y.size == N, 'y and x must have same first dimension'
    
    b_hat = est_ols(y, x)
    
    residual = y - x@b_hat
    SSR = np.sum(residual ** 2)
    SST = np.sum((y-y.mean()) ** 2)
    R2 = 1.0 - SSR/SST

    if robust_se:
        cov, se = robust(x, residual, T)
        sigma2 = np.nan # not a homogeneous parameter with heteroscedasticity
    else: 
        sigma2, cov, se = variance(transform, SSR, x, T)

    # collect output in a dict 
    res = {
        'b_hat':    b_hat,
        'se':       se,
        't':        b_hat / se,
        'cov':      cov,
        'sigma2':   sigma2,
        'R2':       R2, 
    }
    return res

    
def est_ols(y: np.array, x: np.array) -> np.array:
    """Estimates y on x by ordinary least squares, returns coefficents

    Args:
        >> y (np.array): Dependent variable (can be 1D or 2D)
        >> x (np.array): Independent variable (Needs to have ndim == 2)
    
    Returns:
        np.array: Estimated beta coefficients.
    """
    b_hat = la.solve(x.T@x, x.T@y) # equivalent but less efficient: la.inv(x.T@x) @ x.T@y
    return b_hat



def variance( 
        transform: str, 
        SSR: float, 
        x: np.array, 
        T: int
    ) -> tuple:
    """Calculates the covariance and standard errors from the OLS
    estimation.

    Args:
        >> transform (str): Defaults to ''. If the data is transformed in 
        any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation
        >> SSR (float): Sum of squared residuals
        >> x (np.array): Dependent variables from regression
        >> T (int): The number of time periods in x.

    Raises:
        Exception: If invalid transformation is provided, returns
        an error.

    Returns:
        tuple: sigma2, cov, se
            sigma2: error variance 
            cov: (K,K) covariance matrix of estimated parameters 
            se: (K,1) vector of standard errors 
    """

    NT,K = x.shape
    assert transform in ('', 'fd', 'be', 'fe', 're'), f'Transform, "{transform}", not implemented.'

    # degrees of freedom 
    if transform in ('', 'fd', 'be', 're'):
        # just the K parameters we estimate 
        # (note: for FD, x.shape[0] == N*(T-1) because we already deleted the first obs.)
        df = K 
    elif transform == 'fe': 
        N = int(NT/T)
        df = N + K 

    sigma2 = SSR / (NT - df) 
    
    cov = sigma2*la.inv(x.T@x)
    se = np.sqrt(cov.diagonal())
    return sigma2, cov, se


def robust( x: np.array, residual: np.array, T:int) -> tuple:
    '''Calculates the robust variance estimator 

    Args: 
        x: (NT,K) matrix of regressors. Assumes that rows are sorted 
            so that x[:T, :] is regressors for the first individual, 
            and so forth. 
        residual: (NT,1) vector of residuals 
        T: number of time periods. If T==1 or T==None, assumes cross-sectional 
            heteroscedasticity-robust variance estimator
    
    Returns
        tuple: cov, se 
            cov: (K,K) panel-robust covariance matrix 
            se: (K,1) vector of panel-robust standard errors
    '''

    # If only cross sectional, we can use the diagonal.
    if (not T) or (T == 1):
        Ainv = la.inv(x.T@x)
        u2 = residual ** 2
        xTu2 = x.T * u2 
        cov = Ainv @ (xTu2 @ x) @ Ainv
        
        # Alternative formula (very RAM intensive to materialize the (N,N) matrix np.diag(residual**2))
        # cov = (Ainv @ (x.T@np.diag(residual**2)@x) @ Ainv) 
    
    # Else we loop over each individual.
    else:
        NT,K = x.shape
        N = int(NT / T)
        B = np.zeros((K, K)) # initialize 

        for i in range(N):
            idx_i = slice(i*T, (i+1)*T) # index values for individual i 
            Omega = np.outer(residual[idx_i], residual[idx_i]) # (T,T) matrix of outer product of i's residuals 
            B += x[idx_i].T @ Omega @ x[idx_i] # (K,K) contribution 

        Ainv = la.inv(x.T @ x)
        cov = Ainv @ B @ Ainv
    
    se = np.sqrt(np.diag(cov))
    return cov, se

def predict(beta, x): 
    return x @ beta

def print_table(
        labels: tuple,
        results: dict,
        title="Results",
        decimals:int=4,
        _lambda:float=None,
        **kwargs
    ) -> None:
    """Prints a nice looking table, must at least have coefficients, 
    standard errors and t-values. The number of coefficients must be the
    same length as the labels.

    Args:
        >> labels (tuple): Touple with first a label for y, and then a list of 
        labels for x.
        >> results (dict): The results from a regression. Needs to be in a 
        dictionary with at least the following keys:
            'b_hat', 'se', 't', 'R2', 'sigma2'
        >> headers (list, optional): Column headers. Defaults to 
        ["", "Beta", "Se", "t-values"].
        >> title (str, optional): Table title. Defaults to "Results".
    """
    
    # Unpack the labels
    label_y, label_x = labels
    assert isinstance(label_x, list), f'label_x must be a list (second part of the tuple, labels)'
    assert len(label_x) == results['b_hat'].size, f'Number of labels for x should be the same as number of estimated parameters'
    
    # Create table, using the label for x to get a variable's coefficient,
    # standard error and t_value.
    cols = ['b_hat', 'se', 't'] # extract these from results
    result_subset = {k:v for k,v in results.items() if k in cols} # subset the results dictionary to just the items we require
    tab = pd.DataFrame(result_subset, index=label_x)
    
    # Print header
    print(title)
    print(f"Dependent variable: {label_y}\n")
    
    # Print extra statistics of the model.
    print(f"R2 = {results['R2']:.3f}")
    print(f"sigma2 = {results['sigma2']:.3f}")

    # return table (resulting in it being printed unless it is captured as an output variable)
    return tab.round(4)
