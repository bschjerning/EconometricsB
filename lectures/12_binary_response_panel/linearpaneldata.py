import numpy as np
from tabulate import tabulate
from numpy import linalg as la

def estim(df, yvar, xvar, groupvar, method='fe', cov_type='robust', output=True): 
    # Panel structure
    Nobs, k=df[xvar].shape                  # number of observations and explanatory variables 
    n=df[groupvar].unique().size            # number of groups
    T=np.array(df.groupby([groupvar]).size());  # nx1 vector of observation counts for each group

    # "de-meaning" variables
    if method=='fe':
        x = df[xvar] - df[xvar].groupby(df[groupvar]).transform('mean') 
        y = df[yvar] - df[yvar].groupby(df[groupvar]).transform('mean')
        dgf= Nobs - n - k
    elif  method=='pols':
        x = df[xvar] 
        y = df[yvar]
        dgf= Nobs - n - k
    
    # Estimate linear regression model on de-meaned variables
    y=np.array(y).reshape(Nobs, 1)
    x=np.array(x)
    b_hat=la.inv(x.T@x)@ x.T@y
    r = y - x@b_hat
    sigma2 = r.T@r/dgf
    if cov_type=='robust':
        diag = np.zeros((k, k))
        for i in range(0,T.size,1):
            slice_obj = slice(i*T[i], (i+1)*T[i])
            uhat2 = r[slice_obj]@r[slice_obj].T
            diag += x[slice_obj].T @ uhat2 @ x[slice_obj]
        cov=la.inv(x.T@x)@(diag)@la.inv(x.T@x)
    else :
         cov = sigma2*la.inv(x.T@x)
    se = np.sqrt(np.diag(cov)).reshape(-1, 1)
    
    names = ['yvar', 'parnames', 'b_hat', 'se', 'sigma2', 't-values',  'cov', 'Nobs', 'n', 'k']
    results = [yvar, xvar, b_hat, se, sigma2, b_hat/se, cov, Nobs, n, k]
    out=dict(zip(names, results))
    if output:
        if method == 'fe':        
            print('\nSpecification: Linear Fixed Effects Regression\nDep. var. :', yvar, '\n') 
        elif method == 'pols':       
            print('\nSpecification: Pooled OLS Panel Regression\nDep. var. :', yvar, '\n') 
        print_output(out)

    return out
 
def print_output(results): 
    table=({k:results[k] for k in ['parnames','b_hat', 'se', 't-values']})
    print(tabulate(table, headers="keys",floatfmt="10.4f"))
    print('# of groups:      ', results['n'])
    print('# of observations:', results['Nobs'], '\n')
