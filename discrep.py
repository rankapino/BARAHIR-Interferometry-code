#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:04:59 2020

@author: fran parra-rojas
"""

def discrep( U, s, V, b, delta ):
    
    '''
    DISCREP Discrepancy principle criterion for choosing the reg. parameter.
    
    [x_delta,lambda] = discrep(U,s,V,b,delta,x_0)
    [x_delta,lambda] = discrep(U,sm,X,b,delta,x_0)  ,  sm = [sigma,mu]
    
    Least squares minimization with a quadratic inequality constraint:
    min || x - x_0 ||       subject to   || A x - b || <= delta
       min || L (x - x_0) ||   subject to   || A x - b || <= delta
    where x_0 is an initial guess of the solution, and delta is a
    positive constant.  Requires either the compact SVD of A saved as
    U, s, and V, or part of the GSVD of (A,L) saved as U, sm, and X.
    The regularization parameter lambda is also returned.
    
    If delta is a vector, then x_delta is a matrix such that
       x_delta = [ x_delta(1), x_delta(2), ... ] .
    
    If x_0 is not specified, x_0 = 0 is used.
    Reference: V. A. Morozov, "Methods for Solving Incorrectly Posed
    Problems", Springer, 1984; Chapter 26.
    Per Christian Hansen, IMM, August 6, 2007.
    
    Adapted by Francisco Parra-Rojas
    '''
    
    import sys
    import numpy as np
    
    # Initialization.
    m,_ = np.shape(U)
    n,_ = np.shape(V)
    
    if len(s.shape) == 1:    
        p, = np.shape(s)
        ps = 1
    else:
        p, ps = np.shape(s)
     
    
    if type(delta) == float or type(delta) == int:
        ld = 1
        if delta < 0:
            print('Illegal inequality constraint delta')
            sys.exit()
    else:
        ld = len(delta)
        if min(delta) < 0:
            print('Illegal inequality constraint delta')
            sys.exit()
    
    x_delta= np.zeros((n,ld))
    xlambda = np.zeros((ld,1))
    rho = np.zeros((p,1))
    
    
    x_0 = np.zeros((n,1))
    
    if ps == 1:
        omega = V.T @ x_0
    
    else:
        omega = np.linalg.solve(V,x_0)
    
    # Compute residual norms corresponding to CSVD/CGSVD.
    
    beta = U.T @ b
    
    
    if ps == 1:
        delta_0 = np.linalg.norm(b - U @ beta)
        rho[p-1] = delta_0**2

        
        for i in range(p,-1,-1):
            rho[i-2] = rho[i-1] + (beta[i-1] - s[i-1] * omega[i-1])**2
    
    else:
        delta_0 = np.linalg.norm(b - U @ beta)
        rho[0] = delta_0**2
        for i in range(0,p-2):
            rho[i+1] = rho[i] + (beta[i] - s[i,0] * omega[i])**2
    
    # Check input.       
    if type(delta) == float or type(delta) == int:
        if delta < delta_0:
            print('Irrelevant delta < ||(I - U*U'')*b||')
            sys.exit()
    else:
        ld = len(delta)
        if min(delta) < delta_0:
            print('Irrelevant delta < ||(I - U*U'')*b||')
            sys.exit()
        
    # Determine the initial guess via rho-vector, then solve the nonlinear
    # equation || b - A x ||^2 - delta_0^2 = 0 via Newton's method.
    if ps == 1:
        # The standard-form case.
        s2 = s**2
        for k in range(0,ld):
            if delta[k]**2 >= np.linalg.norm(beta - s*omega)**2 + delta_0**2:
                x_delta[:,k] = x_0
            else:
                dummy = np.min(np.abs(rho - delta[k]**2))
                kmin = int(np.where(np.abs(rho - delta[k]**2) == (np.abs(rho - delta[k]**2)).min())[0])
                
                lambda_0 = s[kmin]
                xlambda[k] = newton( lambda_0, delta[k], s, beta, omega, delta_0 )
                e = s/(s2 + xlambda[k]**2)
                f = s * e
                x_delta[:,k] = (V[:,:p] @ (e * beta + (1-f)*omega.T).T).T
    
    # The overdetermined or square genera-form case.
    elif m >= n:
        
        omega = omega[:p]
        gamma = s[:,0]/s[:,1]
        x_u = V[:,p+1:n] @ beta[p+1:n]
        for k in range(0,ld):
            if delta[k]**2 >= np.linalg.norm(beta[:p] - s[:,0]*omega)**2 + delta_0**2:
                x_delta[:,k] = V @ np.concatenate([omega,U[:,p+1:n].T @ b])
            else:
                abs_val = np.abs(rho - delta[k]**2)
                dummy = np.min(abs_val)

                kmin = int(np.where(abs_val == dummy)[0][0])
                
                lambda_0 = gamma[kmin]
                xlambda[k] = newton( lambda_0, delta[k], s, beta[:p], omega, delta_0 )
                e = gamma/(gamma**2 + xlambda[k]**2)
                f = gamma*e
                x_delta[:,k] = (V[:,:p] @ (e * beta[:p]/s[:,1] + (1-f)*s[:,1]*omega.T).T).T + x_u
    
    # The underdetermined general-form case.
    else:
        omega = omega[:p]
        gamma = s[:,0]/s[:,1]
        x_u = V[:,p+1:m] @ beta[p+1:m]
        for k in range(0,ld):
            if delta[k]**2 >= np.linalg.norm(beta[:p] - s[:,0]*omega)**2 + delta_0**2:
                x_delta[:,k] = V @ np.concatenate([omega,U[:,p+1:m].T @ b])
            else:
                dummy = np.min(np.abs(rho - delta[k]**2))
                kmin = int(np.where(np.abs(rho - delta[k]**2) == (np.abs(rho - delta[k]**2)).min())[0])
                
                lambda_0 = gamma[kmin]
                xlambda[k] = newton( lambda_0, delta[k], s, beta[:p], omega, delta_0 )
                e = gamma/(gamma**2 + xlambda[k]**2)
                f = gamma*e
                x_delta[:,k] = V[:,:p] @ (e * beta[:p]/s[:,1] + (1-f)*s[:,1]*omega) + x_u


    return x_delta, xlambda 


def newton( lambda_0, delta, s, beta, omega, delta_0 ):
    
    '''
    NEWTON Newton iteration (utility routine for DISCREP).

    lambda = newton(lambda_0,delta,s,beta,omega,delta_0)
    
    Uses Newton iteration to find the solution lambda to the equation
       || A x_lambda - b || = delta ,
    where x_lambda is the solution defined by Tikhonov regularization.
    
    The initial guess is lambda_0.
    
    The norm || A x_lambda - b || is computed via s, beta, omega and
    delta_0.  Here, s holds either the singular values of A, if L = I,
    or the c,s-pairs of the GSVD of (A,L), if L ~= I.  Moreover,
    beta = U'*b and omega is either V'*x_0 or the first p elements of
    inv(X)*x_0.  Finally, delta_0 is the incompatibility measure.
    Reference: V. A. Morozov, "Methods for Solving Incorrectly Posed
    Problems", Springer, 1984; Chapter 26.
    Per Christian Hansen, IMM, 12/29/97.
    '''
    
    import sys
    import numpy as np
    
    # Set defaults.
    thr = np.sqrt(np.finfo(float).eps) # Relative stopping criterion.
    it_max = 500 # Max number of iterations.

    # Initialization.
    if lambda_0 < 0:
        print('Initial gess lambda_0 must be nonnegative')
        sys.exit()
    
    if len(s.shape) == 1:    
        p, = np.shape(s)
        ps = 1
    else:
        p, ps = np.shape(s)
    
    if ps == 2:
        sigma = s[:,0]
        s = s[:,0]/s[:,1]
    
    s2 = s**2

    # Use Newton's method to solve || b - A x ||^2 - delta^2 = 0.
    # It was found experimentally, that this formulation is superior
    # to the formulation || b - A x ||^(-2) - delta^(-2) = 0.
    ylambda = lambda_0
    step = 1
    it = 0
    

    while np.abs(step) > thr * ylambda and np.abs(step) > thr and it < it_max:
        it += 1
        f = s2/(s2 + ylambda**2)

        if ps == 1:
            r = (1-f) * (beta.T - np.multiply(s.T,omega.T))
            z = f * r
        else:
            r = (1-f) * (beta - np.multiply(sigma.T,omega.T))
            z = f * r        

        step = (ylambda/4) * (r @ r.T + (delta_0 + delta) * (delta_0 - delta))/(z @ r.T)              
        
        ylambda -= step
        
        # If lambda < 0 then restart with smaller initial guess.
        if ylambda < 0:
            ylambda = 0.5 * lambda_0
            lambda_0 = 0.5 * lambda_0
    
    # Terminate with an error if too many iterations.
    if np.abs(step) > thr * ylambda and np.abs(step) > thr:
        print('Max. number of iterations (%1.0i) reached' %it_max)
        sys.exit()
    
    return ylambda
    
    
    