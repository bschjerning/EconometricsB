import pandas as pd
import numpy as np
import numpy.random as random
from numpy import linalg as la
from scipy.optimize import minimize
from tabulate import tabulate
from scipy.stats import norm
import matplotlib.pyplot as plt
import mestim as M


def panel_setup(df, yvar, xvar, groupvar): 
    Nobs, k=df[xvar].shape                     # number of observations and explanatory variables 
    T=np.array(df.groupby(groupvar).size());   # nx1 vector of observation counts for each group
    n=T.size                                   # number of groups

    y=np.array(df[yvar]).reshape(Nobs, 1)
    x=np.array(df[xvar])
        
    return Nobs, k, n, T, y, x 

def simulate(n=100, nT=10, delta=1, rho=0, psi=0, phi_0=0,  phi_y0=0, sigma_a=0, model='probit', rng=random.default_rng(seed=43)): 
    # Model y_it=1(z_it*delta + rho*y_{it-1} + c_i +e_it>0)
    #       c[i]= phi_0 + phi_y0*y_i0 + a_i,  a_i~N(0, sigma_a^2)

    T=np.ones((n))*(nT+1)
    T=T.astype(int)
    Nobs=int(np.sum(T))
    n=T.size
    
    const=np.ones((Nobs))             # constant
    z_it=rng.normal(0, 1, (Nobs))     # exgogenous variable
    y0=1*(rng.normal(0, 1, (n))>0)    # initial conditions 
    c_i=phi_0 + phi_y0*y0 + sigma_a*rng.normal(0, 1, (n))  # unobserved effect

    # shock
    if model == 'probit':
        e_it =rng.normal(0, 1, (Nobs)) 
    elif model == 'logit':
        e_it =rng.logistic(0, 1, (Nobs)) 
        
    # simulate y, endogenous binary variable

    group =np.empty((Nobs))
    period=np.empty((Nobs))
    y_it  =np.empty((Nobs))
    y_i0  =np.empty((Nobs))
    it=0
    for i in range(n):
        for t in range(int(T[i])):
            y_i0[it]=y0[i];
            if t==0:  # initial conditions
                y_it[it]=y0[i];
                group[it]=i
                period[it]=t
            else:
                y_it[it]=1*(z_it[it]*delta +  y_it[it-1]*rho + c_i[i] + e_it[it]> 0);
                group[it]=i
                period[it]=t
            it+=1

    d=dict(zip(['group', 'period', 'y', 'z', 'y0','const'], [group, period, y_it, z_it, y_i0, const] ) )
    df = pd.DataFrame.from_dict(d)
    df = addlag(df, 'y')
    df=df.dropna()
    return df

def pooled(df, yvar, xvar, groupvar, model='probit', cov_type='sandwich',theta0=0, deriv=2): 
    print('Pooled', model)
    Nobs, k, n, T, y, x = panel_setup(df, yvar, xvar, groupvar)
    Qfun     = lambda beta, out:  Q_pooled(y, x, T, beta, model, out)
    if np.isscalar(theta0): 
        theta0=np.zeros((k+1,1)) 
    res=M.estimation(Qfun, theta0=np.zeros((k,1)), deriv=deriv, cov_type=cov_type, parnames=xvar)
    xb, Gx, gx = Qfun(res.theta_hat, out='predict')
    APE=np.mean(gx)*res.theta_hat
    res.update(dict(zip(['yvar', 'xvar', 'Nobs','k', 'n', 'T', 'APE'], [yvar, xvar, Nobs, k, n, T, APE])))
    print_output(res)
    return res

def rand_effect(df, yvar, xvar, groupvar, model='probit', cov_type='Ainv', theta0=0, deriv=1):
    print('Random effects', model)
    Nobs, k, n, T, y, x = panel_setup(df, yvar, xvar, groupvar)
    Qfun     = lambda beta, out:  Q_RE(y, x, T, beta, model, out)
    if np.isscalar(theta0): 
        theta0=np.zeros((k+1,1)) 
        theta0[-1]=1
    res=M.estimation(Qfun, theta0, deriv=deriv, cov_type=cov_type, parnames=xvar+['sigma_a'])
    res.sigma_a=res.theta_hat[-1]
    xb, Gx, gx = Qfun(res.theta_hat/np.sqrt(1+res.sigma_a**2), out='predict')
    APE=np.mean(gx)*res.theta_hat/np.sqrt(1+res.sigma_a**2)
    res.update(dict(zip(['yvar', 'xvar', 'Nobs','k', 'n', 'T', 'APE'], [yvar, xvar, Nobs, k, n, T, APE])))
    print_output(res, ['parnames','theta_hat', 'se', 't-values', 'jac', 'APE'])
    return res

def Q_pooled(y, x, T, beta, model='probit', out='Q'):
    ''' Pooled linear index model for panel data. e.g pooled probit or logit
        y:      Nobs x 1 np.array of binary response data
        x:      Nobs x k np.array of explanatory variables
        T:      n x 1  np.array of containing number of time observations for each group 
        model:  'probit' or 'logit'
        out:    controls what is returned - can be 'predict','Q', 'dQ', 's_i', or 'H'
    '''

    beta=np.array(beta).reshape(-1,1)             # parameters
    n=T.shape[0];                                 # number of groups
    xb=x @ beta                                   # Linear index 
    Gx=G(xb, model)                               # Response probability at x
    gx=g(xb, model)                               # Density at xb
    Gx=np.minimum(np.maximum(Gx,1e-15),1-1e-15)   # Truncating Gx
    if out=='predict':  return xb, Gx, gx         # Return predicted values

    ll_it = np.log(Gx)*y + np.log(1-Gx)*(1-y)     # Nobsx1 vector of log-likelihood contributions for group i at time t
    q_i= - sumby(ll_it, T)                        # nx1 vector of (negative) log-likelihood contributions 
    if out=='Q': return np.mean(q_i);             # Return Q: sample objective function to be minimized - negative of log-likelihood

    # 1st order derivatives
    s_it=gx*x*(y-Gx)/( Gx* (1-Gx))                # NobsxK array of derivatives of ll_it
    s_i = sumby(s_it, T)                          # nxK array of derivatives of ll_i (transposed scores staked over i)
    if out=='s_i': return s_i                     # Return s_i: nxK array with scores
    if out=='dQ':  return -np.mean(s_i, axis=0);  # Return dQ: array of size K derivative of sample objective function

    # 2nd order derivatives
    H=(gx*x).T @(gx*x/(Gx* (1-Gx)))/n             # Analytical Hessian averaged over all groups i
    # H=s_it.T@ s_it/n                            # Alternative: use product of gradient as Hessian approximation
    if out=='H':    return H; 

def Q_RE(y, x, T, theta, model='probit', out='Q', R=20, rng=random.default_rng(seed=11)):
    ''' Pooled linear index model for panel data. e.g pooled probit or logit
        y:      Nobs x 1 np.array of binary response data
        x:      Nobs x k np.array of explanatory variables
        T:      n x 1  np.array of containing number of time observations for each group 
        model:  'probit' or 'logit'
        out:    controls what is returned - can be 'predict','Q', 'dQ', 's_i', or 'H'
    '''
    Nobs, k= x.shape
    n=T.shape[0];                                 # number of groups

    sigma_a=theta[-1]                             # heterogeneity parameter
    beta=np.array(theta[:-1]).reshape(-1,1)       # slope parameters
    
    xb=x @ beta                                   # Linear index 
    gx=g(xb, model)                               # Density at xb
    Gx=G(xb).reshape(-1,1)
    if out=='predict':  return xb, Gx, gx         # Return predicted values

    # compute alpha for used sample objective function at Legendre quadrature nodes
    q,w=quad_xw(R, a=0, b=1)
    q=q.reshape(1,R)
    w=w.reshape(1,R)
    eta=norm.ppf(q);
    alpha=sigma_a*eta

    G_itq=G(xb + alpha, model)                          # Reponse probability at x for each quadrature point (Nobs x R)
    G_itq=np.minimum(np.maximum(G_itq,1e-15),1-1e-15)   # Truncating G
    F_itq = G_itq*y + (1-G_itq)*(1-y)                   # Nobsx1 vector of log-likelihood contributions for group i at time t
    F_iq  = prodby(F_itq,T)                             # product over time, n by R
    Fi = np.sum(F_iq*w, axis=1).reshape(-1,1)           # sum over quad/sims, n by 1 (likelihood contribution for unit i)
    q_i= - np.log(Fi)                                   # nx1 vector of (negative) log-likelihood contributions 
    if out=='Q': return np.mean(q_i);                   # Return Q: sample objective function to be minimized - negative of log-likelihood

    # 1st order derivatives
    g_itq=g(xb + alpha, model)
    s_itq=g_itq*(y-G_itq)/(G_itq*(1-G_itq)) # Nobs x R
    n_p=theta.shape[0]
    s_i=np.empty((n, n_p))                          # nxK+1 array of derivatives of ll_i (transposed scores staked over i)
    for ip in range(n_p-1):
        dw_iq=sumby(s_itq*x[:,ip].reshape(-1,1), T);  # n x R 
        s_i[:,ip]=np.sum(w*F_iq*dw_iq/Fi.reshape(-1,1),axis =1)  # sum over quad/sim points
    dw_iq=sumby(s_itq*eta, T);  # n x R 
    s_i[:,-1]=np.sum(w*F_iq*dw_iq/Fi,axis =1)       
    if out=='s_i': return s_i                       # Return s_i: nxK array with scores
    if out=='dQ':  return -np.mean(s_i, axis=0);    # Return dQ: array of size K derivative of sample objective function

    # 2nd order derivatives
    H=s_i.T@ s_i/n                                  # Alternative: use product of gradient as Hessian approximation
    if out=='H':    return H; 

def quad_xw(n=10, a=-1, b=1):
    '''
    quad_wx: Procedure to compute quadrature weights and nodes to integrate function f([x])
             on interval a, b with n nodes   
             
    parameters
      n:     number legendre nodes
      a, b:  lower, upper integration limits

    outputs
      x  :  1d array with shape (m,1) nodes for x   
      w  :  1d array with shape (m,1)  '''
    
    xi, wi = np.polynomial.legendre.leggauss(n)
    w= (b-a)/2*wi
    x=(xi+1)*(b-a)/2+a
    return x,w

def addlag(df, var, tid='period', t0=0, lg=1):
    ylag=df[var].shift(lg)
    ylag[df[tid]==t0]=np.nan
    df['l' + str(lg)+'.'+var]=ylag
    return df

def prodby(xit, T): 
    nobs, k=xit.shape
    n=T.size
    xi=np.zeros((n,k))
    for i in range(n):
        xi[i,:]=xit[i*T[i]:(i+1)*T[i],:].prod(axis=0)
    return xi

def sumby(xit, T): 
    nobs, k=xit.shape
    n=T.size
    xi=np.zeros((n,k))
    for i in range(n):
        xi[i,:]=xit[i*T[i]:(i+1)*T[i],:].sum(axis=0)
    return xi

def G(z, model='probit'):
    # Index function for probit and logit models
    if model=='probit':
        return norm.cdf(z)
    elif model=='logit':
        return 1/(1+np.exp(-z))

def g(z, model='probit'):
    # Derivative: g(z)=dG(z)/dz
    if model=='probit':
        return norm.pdf(z)
    elif model=='logit':
        z=-np.abs(z) # use transformation to avoid overflow (ok since g is symmetric)
        return np.exp(z)/(1+np.exp(z))**2
        
def print_output(res, cols=['parnames','theta_hat', 'se', 't-values', 'jac', 'APE']): 
    print('Dep. var. :', res['yvar'], '\n') 

    table=({k:res[k] for k in cols})
    print(tabulate(table, headers="keys",floatfmt="10.5f"))
    print('\n# of groups:      :', res['n'])
    print('# of observations :', res['Nobs'])
    print('# log-likelihood. :', - res['Q']*res['n'], '\n')
    print ('Iteration info: %d iterations, %d evaluations of objective, and %d evaluations of gradients' 
        % (res.nit,res.nfev, res.njev))
    print(f"Elapsed time: {res['time']:0.4f} seconds")
    print('')





