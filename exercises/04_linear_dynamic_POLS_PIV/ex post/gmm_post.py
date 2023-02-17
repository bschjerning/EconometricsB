import numpy as np
from numpy import linalg as la
import LinearDynamic as lm


def system_2sls(y: list, x: list, z: list):
    # Intialize som helper variables
    n_rows = z[0].shape[0]
    n_cols = len(z)

    # Initialize the arrays to fill from first_stage loop
    x_predicted = np.zeros((n_rows, n_cols))
    residual_column = np.zeros((n_rows, n_cols))
    for i in range(n_cols):
        x_predicted[:, i], residual_column[:, i] = first_stage(y[i], x[i], z[i])
    
    # Reshape into one column. Since we use 'C' - ordering, the last index
    # changes the fastest. This ensures that the predicted column should
    # have the correct ordering of persons and time periods.
    x_predicted = x_predicted.reshape(-1, 1)
    residual = residual_column.reshape(-1, 1)

    b_hat = lm.est_ols(y[-1], x_predicted)
    residual = y[-1] - x[-1] @ b_hat
    SSR = residual.T @ residual
    sigma2 = SSR/(y[-1].size - x[-1].shape[1])
    cov = sigma2 * la.inv(x[-1].T @ x[-1])
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)
    t_values = b_hat/se
    
    results = [b_hat, se, sigma2, t_values, np.array(0)]
    names = ['b_hat', 'se', 'sigma2', 't_values', 'R2']
    return dict(zip(names, results))


def first_stage(y: np.array, x: np.array, z: np.array):
    b_hat = lm.est_ols(x, z)

    # We use the predictions in the second stage. 
    # The residuals are needed for estimating the variance.
    x_predictions = z @ b_hat
    residuals = y - x_predictions
    return x_predictions.flatten(), residuals.flatten()


def sequential_instruments(x:np.array, T:int):
    """Takes x, and creates the instrument matrix.

    Args:
        >> x (np.array): The instrument vector that we will use to create a new
        instrument matrix that uses all possible instruments each period.
        >> T (int): Number of periods (in the original dataset, before removing
        observations to do first differences or lags). 

    Returns:
        np.array: A (n*(T - 1), k*T*(T - 1)/2) matrix, that has for each individual
        have used all instruments available each time period.
    """

    n = int(x.shape[0]/(T - 1))
    k = x.shape[1]
    Z = np.zeros((n*(T - 1), int(k*T*(T - 1) / 2)))

    # Loop through all persons, and then loop through their time periods.
    # If first time period, use only that as an instrument.
    # Second time period, use the first and this time period as instrument, etc. 
    # Second last time period (T-1)

    # Loop over each individual, we take T-1 steps.
    for i in range(0, n*(T - 1), T - 1):
        # We make some temporary arrays for the current individual
        zi = np.zeros((int(k*T*(T - 1) / 2), T - 1))
        xi = x[i: i + T - 1]

        # j is a help variable on how many instruments we create each period.
        # The first period have 1 iv variable, the next have 2, etc.
        j = 0
        for t in range(1, T):
            zi[j: (j + t), t - 1] = xi[:t].reshape(-1, )
            j += t
        # It was easier to fill the instruments row wise, so we need to transpose
        # the individual matrix before we add it to the main matrix.
        Z[i: i + T - 1] = zi.T
    return Z