import numpy as np
from numpy import linalg as la
from tabulate import tabulate



def estimate( 
        y: np.ndarray, x: np.ndarray, transform='', n=None, t=None
    ) -> list:
    
    b_hat = est_ols(y, x)
    resid = y - x@b_hat
    u_hat = resid@resid.T
    SSR = resid.T@resid
    SST = (y - np.mean(y)).T@(y - np.mean(y))
    R2 = 1 - SSR/SST

    sigma, cov, se = variance(transform, SSR, x, n, t)
    t_values = b_hat/se
    
    names = ['b_hat', 'se', 'sigma', 't_values', 'R2', 'cov']
    results = [b_hat, se, sigma, t_values, R2, cov]
    return dict(zip(names, results))

    
def est_ols( y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return la.inv(x.T@x)@(x.T@y)

def variance( 
        transform: str, 
        SSR: float, 
        x: np.ndarray, 
        n: int,
        t: int
    ) -> tuple :
    
    k = x.shape[1]
    if not n:
        n = x.shape[0]
    
    if not transform:
        sigma = SSR/(n - k)
    elif transform.lower() in ('fe', 'fd'):
        sigma = SSR/(n * (t - 1) - k)
    elif transform.lower() in ('be','re'):
        sigma = SSR/(t * (n - k))
    else:
        raise Exception('Invalid transform provided.')
    
    cov = sigma*la.inv(x.T@x)
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)
    return sigma, cov, se


def print_table(
        labels: tuple,
        results: dict,
        headers=["", "Beta", "Se", "t-values"],
        title="Results",
        **kwargs
    ) -> None:
    label_y, label_x = labels
    # Create table for data on coefficients
    table = []
    for i, name in enumerate(label_x):
        row = [
            name, 
            results.get('b_hat')[i], 
            results.get('se')[i], 
            results.get('t_values')[i]
        ]
        table.append(row)
    
    # Print table
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, **kwargs))
    
    # Print data for model specification
    print(f"R\u00b2 = {results.get('R2').item():.3f}")
    print(f"\u03C3\u00b2 = {results.get('sigma').item():.3f}")
    
    
def perm( Q_T: np.ndarray, A: np.ndarray, t=0) -> np.ndarray:
    """Takes a transformation matrix and performs the transformation on 
    the given vector or matrix.

    Args:
        Q_T (np.array): The transformation matrix. Needs to have the same
        dimensions as number of years a person is in the sample.
        
        A (np.array): The vector or matrix that is to be transformed. Has
        to be a 2d array.

    Returns:
        np.array: Returns the transformed vector or matrix.
    """
    # We can infer t from the shape of the transformation matrix.
    if t==0:
        t = Q_T.shape[1]

    # Initialize the numpy array
    Z = np.array([[]])
    Z = Z.reshape(0, A.shape[1])

    # Loop over the individuals, and permutate their values.
    for i in range(int(A.shape[0]/t)):
        Z = np.vstack((Z, Q_T@A[i*t: (i + 1)*t]))
    return Z
