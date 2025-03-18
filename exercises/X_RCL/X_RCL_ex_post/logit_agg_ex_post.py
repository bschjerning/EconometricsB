import numpy as np
import pandas as pd

import numpy.linalg as la
import linear

def singlechoiceprob(delta_m,theta=None,w=None,v=None,x_random=None):
    if w is None:
        util = delta_m # fill in
        

    else:
        betai = np.vstack((theta[0]*v[:,0], theta[1]*v[:,1]))
        Nw = np.shape(v)[0]
        deltas = np.repeat(np.array(delta_m),Nw).reshape(-1,Nw) 
        sij = np.zeros(shape = (delta_m.size,len(w)))

        mu = x_random @ betai # fill in
        util = deltas + mu # fill in

    exp_util = np.exp(util)
    sj = np.zeros(delta_m.size)
    numerator = exp_util # Jx
    
    if w is None:
            denominator = np.sum(exp_util) + 1  # fill in
            sj = numerator / denominator
            sij=None
    else:
            denominator = np.sum(exp_util,axis=0,keepdims=True) + 1 # fill in
            sij = numerator / denominator
            sj = sij @ w 
    return sj,sij


def iteration(func,x0,max_iter=1e8,tol=1e-8): # brute force iteration, may be slow but seems to get there
    
    # initialise
    i = 0
    xold = x0
    error = 1.0

    while ((i<max_iter) & (error>tol)):

        xnew = func(xold)

        error = np.mean(np.abs(xnew-xold)) # iterates directly on the deltas

        xold = xnew # update
        i = i+1

    if i > max_iter:
        print('max iterations reached, root not necessarily found')

    return xnew

def print_table(
        thtbeta_label: list,
        results: dict,
        error_struct,
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
            'theta', 'se', 't'
        headers (list, optional): Column headers. Defaults to 
            ["", "Beta", "Se", "t-values"].
        title (str, optional): Table title. Defaults to "Results".
        num_decimals: (int) where to round off results (=None to disable)
    """
    if  error_struct == 'vanilla':
        assert len(thtbeta_label) == len(results['beta'])
        t = results['beta']/results['se']
        estimates = results['beta']

    else:
        assert len(thtbeta_label) == len(results['thtbeta'])
        t = results['thtbeta']/results['se']
        estimates = results['thtbeta']
    
    tab = pd.DataFrame({
       'coeff': estimates, 
        'se': results['se'], 
        't-stat': t,
        }, index=thtbeta_label)
    
    if num_decimals is not None: 
        tab = tab.round(num_decimals)
    
    # Print the table
    print(title)
    print('GMM Objective',results['fval'])#.round(4))
    print('Residual Variance',np.var(results['resid']).round(4))
    
    return tab 


def fp_squarem(g,x,maxiter = 1e5,con_tol = 1e-14):

# USAGE: [delta,iter,xchng]=fp_squarem(f,x,optional arguments)
# Applies the SQUAREM acceleration method for fixed point iterations port
# of SQUAREM in R by Ravi Varadhan
# C. Conlon (November 2014)
 
# f is a function handle that performs a single fixed point iteration
# x is an initial guess of the value such that f(x) = x
 
# Optional Arguments

# algorithm: "squarem" (default), "contraction"
# alphaversion: (1, 2, 3) different step-length-schemes (default=3)
# noisy : 0 for no output, 1 for final output only, 2 for iteration level output (default=0)
# con_tol : tolerance for convergence (default 1e-13)
# max_iter: maximum number of iterations before failure (default 1e5)
# stepmin0/stepmax0: initial step min/max (default=1)
# mstep: accepted step scaling factor (default 4).

    #definition of the class starts here  
    class param_fp():  

        #defining constructor  
        def __init__(self):
            self.algorithm = 'squarem' 
            self.alphaversion = 3
            self.mstep = 4
            self.con_tol = con_tol #1e-14
            self.noisy = 0
            self.max_iter = maxiter #1e5
            self.stepmin0 = 1
            self.stepmax0 = 1
            self.KeepUnmatched = True

    p = param_fp()

    iter = 0
    fpevals = 0
    xchng = np.inf 

    stepmin = p.stepmin0
    stepmax = p.stepmax0

    while ((iter < p.max_iter)  & (xchng > p.con_tol)  ):
        iter = iter+1
        x1 = g(x)
        fpevals=fpevals+1        
        
        q1 = x1-x
        x2 = g(x1)
        fpevals = fpevals+1 # function evaluations?
        q2 = x2 - x1
        
        # Form quadratic terms
        sr2 = q1.T @ q1
        qdiff = q2-q1
        sv2 = qdiff.T @ qdiff

        srv = q1.T @ qdiff
        
        # Get the step-size
        alpha = compute_alpha(sv2,sr2,srv,stepmin,stepmax,p.alphaversion)
        xtmp = x + 2 * alpha * q1+ alpha**2 * (q2-q1) # NB check whether this needs matrix mult instead
        
        # Fixed point iteration beyond the quadratic step
        xnew = g(xtmp)
        fpevals = fpevals+1
        
        if(any(np.isnan(xnew))):
            xnew = iteration(g,x2)
            #print('Error')
            #xnew = x2
        else:
            None
        
        if alpha == stepmax:
            stepmax = p.mstep * stepmax
        
        elif alpha == stepmin and alpha < 0:
            stepmin = p.mstep * stepmin
        else:
            None

        if(p.noisy==2): # when would this ever happen?? we set it to zero above
            print(' Iteration ', str(iter) ,' delta change: ', str(la.norm(xnew-xtmp,ord='inf')),
                ' and alpha=', str(alpha),' ( ', str(stepmin), ' , ' ,str(stepmax) ,')')
        else:
            None

        xchng = np.mean(abs(xnew-xtmp))
        x = xnew
    
    return xnew,iter,xchng

def compute_alpha(sv2,sr2,srv,stepmin,stepmax,alphaversion):

    # sv2 akin to second derivative, f'' (rate of accelaration/change in slope)
    # sr2 akin to first derivative, f' (slope)

    if sv2 == 0.0:
        sv2 = 1.0 # overwrite to avoid division by zero below

    if alphaversion == 1:
        alpha = -srv/sv2
    elif alphaversion == 2:
        alpha = -sr2/srv
    else:
        alpha = np.sqrt(sr2/sv2)
        
    alpha = max(stepmin,min(stepmax,alpha))

    return alpha

def getcovar(dhat_final,X,Z,W,v,w,X_random,mktids,thetahat):

    w=w.flatten()

    # get jacobians

    Jac = np.zeros ((len(dhat_final),2))

    for m in np.unique(mktids):

        delta_m = dhat_final[mktids==m].flatten()
        x_random_m = X_random[mktids==m]
        
        sjt,sijt = singlechoiceprob(delta_m,thetahat,w,v,x_random_m) 
        Jac_d = -sijt @ np.diag(w) @ sijt.T + np.diag(sjt)

        betamask = np.array([[1,1],[2,2]])

        nK = np.shape(betamask)[0] 
        Jac_theta = np.zeros((len(delta_m),nK))

        for kk in range(nK):
            xindex = betamask[kk,0]-1
            vindex = betamask[kk,1]-1 # bc zero indexing in numpy (I think)

            Jac_theta[:,kk] = JacSigma(xindex,vindex,x_random_m,w,v,sijt)
        
        Jac[mktids==m,:] = la.solve(np.diag(1/sjt) @ Jac_d,np.diag(1/sjt) @ Jac_theta)

    # put it all together to get the standard errors

    NT = Z.shape[0]

    _,uhat = linear.est_piv(dhat_final,X,Z,W)

    g = uhat[:,np.newaxis]*Z
    gstar = (g-np.mean(g))/NT
    S = gstar.T@gstar
    G = Z.T @ np.hstack((Jac,X))/NT
            
    W2 = la.inv(NT**2*S)
    se = np.sqrt(np.diag(la.inv(G.T @ W @ G) @ G.T @ W @ S @ W @ G @ la.inv(G.T @ W @ G)))

    return se

def JacSigma(xindex,vindex,xrandom, w,v,pijt): 

    vk = v[:,vindex]
    
    xk = xrandom[:,xindex]

    a = pijt*vk.T
    b = xk[:,np.newaxis]-(xk.T@pijt)[np.newaxis,:]
    c = a*b
    
    JacVec = c @ w 

    return JacVec

