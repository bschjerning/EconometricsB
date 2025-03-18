import numpy as np
from numpy import linalg as la
from tabulate import tabulate



def estimate( 
        y: np.ndarray, x: np.ndarray,z=None, transform='', T:int=None,robust_se=False
    ) -> list:
    """Uses the provided estimator (mostly OLS for now, and therefore we do 
    not need to provide the estimator) to perform a regression of y on x, 
    and provides all other necessary statistics such as standard errors, 
    t-values etc.  

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)
        >> transform (str, optional): Defaults to ''. If the data is 
        transformed in any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation.
        >>t (int, optional): If panel data, t is the number of time periods in
        the panel, and is used for estimating the variance. Defaults to None.

    Returns:
        list: Returns a dictionary with the following variables:
        'b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov'
    """
    if z is None:
        b_hat = est_ols(y, x)  # Estimated coefficients
        residual = y - x@b_hat  # Calculated residuals
    else:
        b_hat,residual = est_piv(y,x,z)

    SSR = residual.T@residual  # Sum of squared residuals
    SST = (y - np.mean(y)).T@(y - np.mean(y))  # Total sum of squares
    R2 = 1 - SSR/SST

    sigma2, cov, se = variance(transform, SSR, x, T)
    if robust_se:
        cov, se = robust(x, residual, T)
    t_values = b_hat.flatten()/se.flatten()
    
    names = ['b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov']
    results = [b_hat, se, sigma2, t_values, R2, cov]
    return dict(zip(names, results))

    
def est_ols( y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Estimates y on x by ordinary least squares, returns coefficents

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)

    Returns:
        np.array: Estimated beta coefficients.
    """
    return la.inv(x.T@x) @ (x.T@y)

def est_piv( y, x, z,w=None):
    if w is None:
        w = la.inv(z.T@z)
    
    betahat = la.solve((x.T@z @ w @ z.T @ x), (x.T @ z @ w @ z.T@y))

    resid = y - x@betahat
    
    return betahat,resid

def variance( 
        transform: str, 
        SSR: float, 
        x: np.ndarray, 
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
        >> x (np.ndarray): Dependent variables from regression
        >> t (int): The number of time periods in x.

    Raises:
        Exception: If invalid transformation is provided, returns
        an error.

    Returns:
        tuple: Returns the error variance (mean square error), 
        covariance matrix and standard errors.
    """

    # Store n and k, used for DF adjustments.
    k = x.shape[1]
    if transform in ('', 'fd', 'be'):
        n = x.shape[0]
    else:
        n = x.shape[0]/T

    # Calculate sigma2
    if transform in ('', 'fd', 'be'):
        sigma2 = (np.array(SSR/(n - k)))
    elif transform.lower() == 'fe':
        sigma2 = np.array(SSR/(n * (T - 1) - k))
    elif transform.lower() == 're':
        sigma2 = np.array(SSR/(T * n - k))
    else:
        raise Exception('Invalid transform provided.')
    
    cov = sigma2*la.inv(x.T@x)
    se = np.sqrt(cov.diagonal())
    return sigma2, cov, se


def print_table(
        labels: tuple,
        results: dict,
        headers=["", "Beta", "Se", "t-values"],
        title="Results",
        _lambda:float=None,
        **kwargs
    ) -> None:
    """Prints a nice looking table, must at least have coefficients, 
    standard errors and T-values. The number of coefficients must be the
    same length as the labels.

    Args:
        >> labels (tuple): Touple with first a label for y, and then a list of 
        labels for x.
        >> results (dict): The results from a regression. Needs to be in a 
        dictionary with at least the following keys:
            'b_hat', 'se', 't_values', 'R2', 'sigma2'
        >> headers (list, optional): Column headers. Defaults to 
        ["", "Beta", "Se", "T-values"].
        >> title (str, optional): Table title. Defaults to "Results".
        _lambda (float, optional): Only used with Random effects. 
        Defaults to None.
    """
    
    # Unpack the labels
    label_y, label_x = labels
    
    # Create table, using the label for x to get a variable's coefficient,
    # standard error and t_value.
    table = []
    for i, name in enumerate(label_x):
        row = [
            name, 
            results.get('b_hat')[i], 
            results.get('se')[i], 
            results.get('t_values')[i]
        ]
        table.append(row)
    
    # Print the table
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, **kwargs))
    
    # Print extra statistics of the model.
    print(f"R\u00b2 = {results.get('R2').item():.3f}")
    print(f"\u03C3\u00b2 = {results.get('sigma2').item():.3f}")
    if _lambda: 
        print(f'\u03bb = {_lambda.item():.3f}')



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
        uhat2 = residual ** 2
        uhat2_x = uhat2 * x # elementwise multiplication: avoids forming the diagonal matrix (RAM intensive!)
        cov = Ainv @ (x.T@uhat2_x) @ Ainv
    
    # Else we loop over each individual.
    else:
        NT,K = x.shape
        N = int(NT / T)
        B = np.zeros((K, K)) # initialize 

        for i in range(N):
            idx_i = slice(i*T, (i+1)*T) # index values for individual i 
            Omega = residual[idx_i]@residual[idx_i].T # (T,T) matrix of outer product of i's residuals 
            B += x[idx_i].T @ Omega @ x[idx_i] # (K,K) contribution 

        Ainv = la.inv(x.T @ x)
        cov = Ainv @ B @ Ainv
    
    se = np.sqrt(np.diag(cov)).reshape(-1, 1)
    return cov, se