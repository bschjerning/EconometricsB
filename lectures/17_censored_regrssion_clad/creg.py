# Selection of routines for estimation of censored regression models

import numpy as np
import pandas as pd
import mestim as M
from tabulate import tabulate
from numpy import linalg as la
from numpy import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

def sim_data(N=500, beta=[1,1], sigma=1, delta=0, gamma=0, error_type='normal'):
    beta=np.array(beta).reshape(-1,1)
    x =  random.random(size=(N, beta.shape[0]))
    x[:,0]=np.ones((N))
    u=sigma*sim_error(N, x, error_type, delta) # error has mean 0 and std.dev sigma 
    c=gamma*sim_error(N, x, 'normal', 0) # error has mean 0 and std.dev sigma 
    ys = x @ beta + u + gamma*c
    y = np.maximum(ys, 0)

    N, K, lbl_params=labels(x)
    xvars=lbl_params[:-1]; 
    varnames=columns=['y'] + xvars +['ys', 'u']
    df = pd.DataFrame(data=np.hstack((y, x, ys, u)), columns=varnames )
    return df, xvars

def sim_error(N, x, error_type, delta): 

    sigma_i=np.exp(delta*x[:,-1]).reshape(N,1)
    # Generate errors
    if error_type=='normal':
        u=random.normal(0, 1, size=(N,1))
    elif error_type=='log-normal': 
        u=np.exp(random.normal(0, 1, size=(N,1)));    
    elif error_type=='uniform': 
        u=random.random(size=(N,1)); 
    elif error_type=='mix_sym': 
        # Parameter for mixture of two normals  
        p=0.5                       # Mixing parameter    
        mu1=-2;mu2=2               # Means    
        se1=1; se2=1;               # Standard deviation
        u_u=random.random(size=(N,1))
        u1=random.normal(size=(N,1))
        u2=random.normal(size=(N,1))
        u=(mu1+se1*u1)*(u_u<=p) + (mu2+se2*u2)*(u_u>p)
    elif error_type=='mix_asym': 
        # Parameter for mixture of two normals  
        p=0.5                       # Mixing parameter    
        mu1=-2;mu2=2               # Means    
        se1=2; se2=1;               # Standard deviation
        u_u=random.random(size=(N,1))
        u1=random.normal(size=(N,1))
        u2=random.normal(size=(N,1))
        u=(mu1+se1*u1)*(u_u<=p) + (mu2+se2*u2)*(u_u>p)

    u=sigma_i*u;
    u=u-np.mean(u);
    u=u/np.std(u);
    return u

def tobit(yvar, xvars, cov_type='Ainv',theta0=None, deriv=1, quiet=False): 
    # Objective function and derivatives for 
    x=np.array(xvars)
    N, K, xvars=labels(x)
    y=np.array(yvar).reshape(N,1)
    Qfun     = lambda theta, out:  Q_tobit(theta, y, x, out)
    if theta0 is None: 
        theta0=np.zeros((K+1,1)) 
        b=la.inv(x.T@x)@x.T@y; 
        theta0[0:-1,:]=b
        theta0[-1,:]=np.sqrt(np.mean((y-x@b)**2))
        
    res=M.estimation(Qfun, theta0, deriv, cov_type, parnames=xvars)   
    res.update(dict(zip(['yvar', 'xvars', 'N','K', 'n'], ['y', xvars, N, K, N])))
    if quiet==False:    
        print('Tobit model')
        print('Fractions of observations that are censored: ', np.mean(1*(y==0)))
        # print('Initial log-likelihood', -Qfun(theta0, 'Q'))
        # print('Initial gradient\n', -Qfun(theta0, 'dQ'))
        print_output(res)
    return res

def clad_ilpa(yvar, xvars, maxiter_ilpa=20, quiet=False, b=None):  
    x=np.array(xvars)
    N, K, xvars=labels(x)
    y=np.array(yvar).reshape(N,1)

    if np.mean(1*(y>0))<0.5:
        print('Error: More than half of observations are censored. Beta is unidentified')
        return np.nan*np.ones((K,1))

    if b is None: # use lad estimates as starating values
        lad=sm.QuantReg(y,x).fit(q=0.5, p_tol=1e-05)
        b=np.array(lad.params).reshape(-1,1)

    for i in range(maxiter_ilpa):
        b0=b;
        yhat=x@b
        lad=sm.QuantReg(y[yhat>0],x[yhat[:,0]>0,:]).fit(q=0.5,p_tol=1e-05)
        b=np.array(lad.params).reshape(-1,1)

        if (b==b0).all():  # "convrgence achieved";
            if quiet==False:
                print('CLAD finished in %d iterations using ILPA' %i)
                print('Fractions of observations that are censored: ', np.mean(1*(y==0)))
                print(lad.summary())    
            return np.array(b).reshape(-1,1)
    
    return np.nan*np.ones((K,1))

def Q_tobit(theta, y, x, out='Q'): # sample objective function and derivatives for tobit
    N, K = x.shape 
    y=np.array(y).reshape(N,1);
    theta=np.array(theta).reshape(K+1,1)
    beta=theta[:-1,:]
    sigma=theta[-1,:]
    Phi_i=norm.cdf(x@beta/sigma)
    Phi_i=np.minimum(np.maximum(Phi_i,1e-15),1-1e-15) 
    ll_i = 1*(y == 0)*np.log(1-Phi_i)  -  1*(y > 0)*((y-x@beta)**2/(2*sigma**2) + np.log(sigma**2)/2)
    if out=='Q': return -np.mean(ll_i, axis=0)

    s_i=s_i_tobit(beta, sigma, y, x, Phi_i) # computes NxK array with scores 
    if out=='s_i': return s_i  # Return s_i
    if out=='dQ':  return -np.mean(s_i, axis=0);  # Return dQ: array of size K derivative of sample objective function

def s_i_tobit(beta, sigma, y, x, Phi_i): 
    # derivatives 
    phi_i=norm.pdf(x@beta/sigma)
    phi_i=np.minimum(np.maximum(phi_i,1e-15),1-1e-15) 
    s_i_beta= 1/sigma**2*((y > 0)*(y-x@beta) - 1*(y == 0)*sigma*phi_i/(1-Phi_i))*x 
    s_i_sigma =  1*(y == 0)*phi_i/(1-Phi_i)*x@beta/(sigma**2) 
    s_i_sigma += 1*(y > 0)*((y-x@beta)**2/(sigma**3) -1/(sigma))
    s_i=np.append(s_i_beta, s_i_sigma.reshape(-1,1),  axis=1)
    return s_i

def scatter_ols(x,y,ys=None):
    x=x[:,1].reshape(-1,1)

    plt.figure(figsize=(10,8)) 
    plt.scatter(x[y==0], y[y==0], s=0.4, label='y=0')
    plt.scatter(x[y>0], y[y>0],  s=0.4, label='y>0')
    if not ys is None:
        plt.scatter(x[ys<0], ys[ys<0], s=0.4, label='y<0')
    a, b = np.polyfit(x[:,0], y[:,0], 1)
    plt.plot(x, a*x + b, label='OLS (observed, y)')
    a, b = np.polyfit(x[y>0], y[y>0], 1)
    plt.plot(x, a*x + b, label='OLS (truncated, y>0)')
    if not ys is None:
        a, b = np.polyfit(x[:,0], ys[:,0], 1)
        plt.plot(x, a*x + b, label='OLS (latent variable, y*)')
    plt.legend(loc="upper left", prop={'size': 15})
    plt.show()

def condmean(beta, sigma, y, x):
    xb=x@beta                   # E(y*|x)
    Phi = norm.cdf(xb/sigma)    # P(y=0|x)
    phi = norm.pdf(xb/sigma)    
    inv_mills = sigma*phi/Phi   # inverse mills ratio
    Ey_trunc= xb + inv_mills    # E[y|x,y>0]
    Ey= xb*Phi + sigma*phi      # E[y|x]
    return xb, Phi, phi, inv_mills, Ey_trunc, Ey

def scatter_condmean(beta, sigma, x,y, ys=None):
    y=y[x[:,1].argsort()]
    x=x[x[:,1].argsort()]
    x1=x[:,1].reshape(-1,1)
    beta=np.array(beta).reshape(-1,1)
    
    xb, Phi, phi, inv_mills, Ey_trunc, Ey = condmean(beta, sigma, y, x)

    plt.figure(figsize=(10,8)) 
    plt.scatter(x1[y==0], y[y==0], s=0.4, label='y=0')
    plt.scatter(x1[y>0], y[y>0],  s=0.4, label='y>0')
    if not ys is None:
        plt.scatter(x1[ys<0], ys[ys<0], s=0.4, label='y<0')
    plt.plot(x1, xb, label='E(y*|x)=x*beta')
    a, b = np.polyfit(x1[y>0], y[y>0], 1)
    plt.plot(x1, Ey, label='E(y|x)')
    plt.plot(x1, Ey_trunc, label='E(y|x, y>0)')
    plt.legend(loc="upper left", prop={'size': 15})
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
    # print ('Iteration info: %d iterations, %d evaluations of objective, and %d evaluations of gradients' 
    #     % (res.nit,res.nfev, res.njev))
    # print(f"Elapsed time: {res['time']:0.4f} seconds")
    print('')

def plot_Qn(Qn, theta, k=0, interval=[-10,10], title=''):
    # plot objective function Qn(theta) against parameter k over grid for theta[k]
    grid=np.linspace(interval[0],interval[1], 200)
    Q=np.empty(grid.shape)
    theta_i=theta.copy()
    for i in range(grid.shape[0]):
        theta_i[k]=grid[i]
        Q[i]=Qn(theta_i)
    fig=plt.plot(grid, Q)
    plt.axvline(x=theta[k], color='k', linestyle='--')
    plt.title(title)
    plt.xlabel('theta[' + str(k)+ '] = ' + str(theta[k]))

