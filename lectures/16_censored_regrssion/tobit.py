import numpy as np
import mestim as M
from tabulate import tabulate
from numpy import linalg as la
from numpy import random
import matplotlib.pyplot as plt
from scipy.stats import norm

def tobit(y, x, cov_type='Ainv', theta0=None, deriv=1, quiet=False):
    """
    Estimates a Tobit model via maximum likelihood.

    Model Specification:
        y_i^* = x_i β + u_i
        y_i = max(0, y_i^*)
        u_i ~ N(0, sigma^2)

    Parameters:
    - y (array): Observed dependent variable (censored at zero).
    - x (array): Independent variables.
    - cov_type (str): Type of covariance estimator ('Ainv', 'sandwich', etc.).
    - theta0 (array or None): Initial parameter guess (if None, estimated).
    - deriv (int): Order of derivatives used (1 for gradient, 2 if Hessian is available).
    - quiet (bool): If True, suppresses printed output.

    Returns:
    - res (dict): Dictionary containing estimates, standard errors, and diagnostics.
    """
    # Define objective function
    Qfun = lambda theta, out: Q_tobit(theta, y, x, out)

    # Get variable labels and dimensions
    N, K, xvars = labels(x)

    # Initialize theta0 if not provided
    if theta0 is None:
        theta0 = np.zeros((K + 1, 1))
        b = la.inv(x.T @ x) @ x.T @ y  # initialize with OLS estimate
        theta0[:-1, :] = b
        theta0[-1, :] = max(np.std(y - x @ b), 1e-6)  # Ensure sigma is positive

    # Run M-estimation using the estimation function
    res = M.estimation(Qfun, theta0, deriv, cov_type, parnames=xvars)

    # Store metadata in result dictionary
    res.update(dict(zip(['yvar', 'xvars', 'N', 'K', 'n'], ['y', xvars, N, K, N])))

    # Print results if not in quiet mode
    if not quiet:
        print("Tobit Model")
        print(f"Fraction of censored observations: {np.mean(y == 0):.4f}")
        print(f"Initial log-likelihood: {-Qfun(theta0, 'Q'):.4f}")
        print(f"Initial gradient norm: {la.norm(-Qfun(theta0, 'dQ')):.4f}")
        print_output(res)

    return res

def sim_data(N=500, beta=[1, 1], sigma=1, error_type='normal'):
    """
    Simulates data for a Tobit model with a corner solution at zero.

    Model Specification:
        y_i^* = x_i β + u_i
        y_i = max(0, y_i^*)
        u_i ~ F(0, sigma), where F is the chosen error distribution.

    Parameters:
    - N (int): Number of observations.
    - beta (list or array): True regression coefficients (including intercept).
    - sigma (float): Standard deviation of the error term.
    - error_type (str): Type of error distribution. Options:
        - 'normal': Standard normal distribution.
        - 'log-normal': Log-normal distribution.
        - 'uniform': Uniform(0,1).
        - 'het': Heteroskedastic errors (variance depends on x).
        - 'mix_sym': Mixture of two symmetric normals.
        - 'mix_asym': Mixture of two asymmetric normals.

    Returns:
    - y (array): Observed dependent variable (censored at zero).
    - ys (array): Latent variable (before censoring).
    - x (array): Independent variables (including an intercept).
    - u (array): Error term (zero mean, unit variance).
    """
    # Ensure beta is a column vector
    beta = np.array(beta).reshape(-1, 1)
    
    # Generate independent variables (X), first column is an intercept
    x = random.normal(size=(N, beta.shape[0]))  
    x[:, 0] = 1  # Set first column to 1 for the intercept
    
    # Generate error term u with the specified distribution
    u = sigma * sim_error(N, x, error_type)  # Error has mean 0, std dev sigma
    
    # Compute latent variable y*
    ys = x @ beta + u  
    
    # Apply censoring: y = max(y*, 0)
    y = np.maximum(ys, 0)  
    
    return y, ys, x, u

def sim_error(N, x, error_type='normal'):
    """
    Generates error terms for simulated data in a Tobit model.

    Parameters:
    - N (int): Number of observations.
    - x (array): Independent variables (used for heteroskedastic errors).
    - error_type (str): Distribution of the error term. Options:
        - 'normal': Standard normal errors.
        - 'log-normal': Log-normal errors.
        - 'uniform': Uniform(0,1) errors.
        - 'het': Heteroskedastic errors (variance depends on x).
        - 'mix_sym': Mixture of two symmetric normal distributions.
        - 'mix_asym': Mixture of two asymmetric normal distributions.

    Returns:
    - u (array): Error term (zero mean, unit variance).
    """
    if error_type == 'normal':
        u = random.normal(0, 1, size=(N, 1))  # Standard normal errors
    elif error_type == 'log-normal':
        u = np.exp(random.normal(0, 1, size=(N, 1)))  # Log-normal errors
    elif error_type == 'uniform':
        u = random.random(size=(N, 1))  # Uniform(0,1) errors
    elif error_type == 'het':  # Heteroskedastic errors (variance depends on x)
        u = (0.2 * x[:, 1] ** 2).reshape(-1, 1) * random.normal(0, 1, size=(N, 1))
    elif error_type in ['mix_sym', 'mix_asym']:  # Mixture of two normal distributions
        p = 0.5  # Mixing probability
        mu1, mu2 = -2, 2  # Means
        se1, se2 = (1, 1) if error_type == 'mix_sym' else (2, 1)  # Std deviations
        u = np.where(random.random(size=(N, 1)) <= p,
                     mu1 + se1 * random.normal(size=(N, 1)),
                     mu2 + se2 * random.normal(size=(N, 1)))

    # Normalize errors to have mean 0 and std dev 1
    u = (u - np.mean(u)) / np.std(u)
    return u

def Q_tobit(theta, y, x, out='Q'):
    """
    Computes the Tobit log-likelihood function and its derivatives.

    Model Specification:
        y_i^* = x_i β + u_i
        y_i = max(0, y_i^*)
        u_i ~ N(0, sigma^2)

    Parameters:
    - theta (array): Model parameters [beta, sigma].
    - y (array): Observed dependent variable (censored at zero).
    - x (array): Independent variables.
    - out (str): Specifies output:
        - 'Q' (default): Returns negative log-likelihood (to minimize).
        - 'dQ': Returns gradient vector (score function).
        - 's_i': Returns individual score contributions.

    Returns:
    - Depends on `out`: Negative log-likelihood, gradient, or scores.
    """
    N, K = x.shape
    theta = np.array(theta).reshape(K+1, 1)  # Ensure column vector
    beta, sigma = theta[:-1, :], theta[-1, :]

    # Compute CDF of latent variable for censored observations
    Phi_i = norm.cdf(x @ beta / sigma)
    Phi_i = np.clip(Phi_i, 1e-12, 1 - 1e-12)  # Avoid log(0) issues

    # Compute log-likelihood (each observation)
    ll_i = (y == 0) * np.log(1 - Phi_i) - (y > 0) * ((y - x @ beta)**2 / (2 * sigma**2) + np.log(sigma**2) / 2)

    if out == 'Q':
        return -np.mean(ll_i)  # Return negative log-likelihood

    # Compute score function
    s_i = s_i_tobit(beta, sigma, y, x, Phi_i)

    if out == 's_i':
        return s_i  # Return individual score contributions
    elif out == 'dQ':
        return -np.mean(s_i, axis=0)  # Return gradient vector (negative mean score)
    
def s_i_tobit(beta, sigma, y, x, Phi_i):
    """
    Computes individual score contributions for the Tobit model.

    Model Specification:
        y_i^* = x_i β + u_i
        y_i = max(0, y_i^*)
        u_i ~ N(0, sigma^2)

    Parameters:
    - beta (array): Regression coefficients.
    - sigma (float): Standard deviation of the error term.
    - y (array): Observed dependent variable (censored at zero).
    - x (array): Independent variables.
    - Phi_i (array): CDF of latent variable (used for censored observations).

    Returns:
    - s_i (array): (N, K+1) matrix of individual score contributions.
    """
    phi_i = norm.pdf(x @ beta / sigma)
    phi_i = np.clip(phi_i, 1e-12, 1 - 1e-12)  # Avoid division errors

    # Score contributions for beta (K parameters)
    s_i_beta = (1 / sigma**2) * ((y > 0) * (y - x @ beta) - (y == 0) * sigma * phi_i / (1 - Phi_i)) * x

    # Score contribution for sigma
    s_i_sigma = (y == 0) * (phi_i / (1 - Phi_i)) * (x @ beta / sigma**2)  # Censored observations
    s_i_sigma += (y > 0) * ((y - x @ beta)**2 / sigma**3 - 1 / sigma)  # Uncensored observations

    # Combine score contributions into a single array
    s_i = np.append(s_i_beta, s_i_sigma.reshape(-1, 1), axis=1)

    return s_i

import matplotlib.pyplot as plt
import numpy as np

def scatter_ols(x, y, ys=None):
    """
    Plots OLS regression lines for observed and latent variables in a Tobit model.

    Parameters:
    - x (array): Independent variable (assumes second column of input matrix).
    - y (array): Observed dependent variable (censored at zero).
    - ys (array, optional): Latent variable before censoring (y*), if available.

    Returns:
    - None (displays the plot).
    """
    x = x[:, 1].reshape(-1,1)  # Extract the second column and ensure 1D array

    plt.figure(figsize=(10, 6))
    
    # Scatter plot with larger markers for clarity
    plt.scatter(x[y == 0], y[y == 0], s=10, label='y = 0 (Censored)', color='black', alpha=0.7, edgecolors='white')
    plt.scatter(x[y > 0], y[y > 0], s=10, label='y > 0 (Uncensored)', color='blue', alpha=0.7, edgecolors='white')

    if ys is not None:
        plt.scatter(x[ys < 0], ys[ys < 0], s=10, label='y* < 0 (Latent)', color='green', alpha=0.7, edgecolors='white')

    # Fit and plot OLS lines
    a, b = np.polyfit(x.flatten(), y.flatten(), 1)
    plt.plot(np.sort(x), np.sort(a * x + b), label='OLS (Observed, y)', color='blue', linewidth=2)

    a, b = np.polyfit(x[y > 0].flatten(), y[y > 0].flatten(), 1)
    plt.plot(np.sort(x), np.sort(a * x + b), label='OLS (Truncated, y > 0)', color='red', linewidth=2)

    if ys is not None:
        a, b = np.polyfit(x.flatten(), ys.flatten(), 1)
        plt.plot(np.sort(x), np.sort(a * x + b), label='OLS (Latent, y*)', color='green', linewidth=2)

    # Improve plot aesthetics
    plt.xlabel("Independent Variable (x)", fontsize=14)
    plt.ylabel("Dependent Variable (y)", fontsize=14)
    plt.title("OLS Regression with Censored Data", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Move legend outside the plot for better readability
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12, frameon=True)
    plt.tight_layout()  # Adjust layout to fit legend nicely
    plt.show()

def condmean(beta, sigma, y, x):
    xb=x@beta                   # E(y*|x)
    Phi = norm.cdf(xb/sigma)    # P(y=0|x)
    phi = norm.pdf(xb/sigma)    # pdf of y*|x
    inv_mills = sigma*phi/Phi   # inverse mills ratio
    Ey_trunc= xb + inv_mills    # E[y|x,y>0]
    Ey= xb*Phi + sigma*phi      # E[y|x]
    return xb, Phi, phi, inv_mills, Ey_trunc, Ey

def scatter_condmean(beta, sigma, x, y, ys=None):
    """
    Plots conditional expectations and OLS regression lines for the Tobit model.

    Parameters:
    - beta (array): Regression coefficients.
    - sigma (float): Standard deviation of the error term.
    - x (array): Independent variables.
    - y (array): Observed dependent variable.
    - ys (array, optional): Latent variable before censoring (y*), if available.

    Returns:
    - None (displays the plot).
    """
    beta = np.array(beta).reshape(-1, 1)  # Ensure beta is a column vector
    
    # Sort data for proper visualization
    sort_idx = np.argsort(x[:, 1])
    x, y = x[sort_idx], y[sort_idx]
    x1 = x[:, 1].reshape(-1, 1)  # Extract sorted independent variable

    # Compute conditional expectations
    xb, Phi, phi, inv_mills, Ey_trunc, Ey = condmean(beta, sigma, y, x)

    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Scatter plots for observed data
    plt.scatter(x1[y == 0], y[y == 0], s=10, label='y = 0 (Censored)', color='black', alpha=0.7, edgecolors='white')
    plt.scatter(x1[y > 0], y[y > 0], s=10, label='y > 0 (Uncensored)', color='blue', alpha=0.7, edgecolors='white')

    if ys is not None:
        plt.scatter(x1[ys < 0], ys[ys < 0], s=10, label='y* < 0 (Latent)', color='green', alpha=0.7, edgecolors='white')

    # Plot conditional expectations
    plt.plot(x1, xb, label='E(y* | x) = xβ', color='purple', linewidth=2, linestyle='dashed')
    plt.plot(x1, Ey, label='E(y | x)', color='blue', linewidth=2)
    plt.plot(x1, Ey_trunc, label='E(y | x, y > 0)', color='red', linewidth=2)

    # Improve plot aesthetics
    plt.xlabel("Independent Variable (x)", fontsize=14)
    plt.ylabel("Dependent Variable (y)", fontsize=14)
    plt.title("Conditional Expectations in the Tobit Model", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Move legend outside the plot for better readability
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12, frameon=True)
    plt.tight_layout()  # Adjust layout to fit legend nicely
    plt.show()

def labels(x):
    # labels and dimensions for plotting
    N, K = x.shape
    xvars=['x' + str(i)  for i in range(K+1)]
    xvars[-1]='sigma'
    return N, K, xvars

def print_output(res, cols=['parnames','theta_hat', 'se', 't-values', 'jac']): 
    print('Dep. var. :', res['yvar'], '\n') 

    table=({k:res[k] for k in cols})
    print(tabulate(table, headers="keys",floatfmt="10.5f"))
    # print('\n# of groups:      :', res['n'])
    print('# of observations :', res['N'])
    print('# log-likelihood. :', - res['Q']*res['n'], '\n')
    print ('Iteration info: %d iterations, %d evaluations of objective, and %d evaluations of gradients' 
        % (res.nit,res.nfev, res.njev))
    print(f"Elapsed time: {res['time']:0.4f} seconds")
    print('')

