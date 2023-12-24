#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:43:24 2020

@author: fran parra-rojas
"""

def tikhonov(U,s,V,b,xlambda):
    '''
    TIKHONOV Tikhonov regularization.
    
    [x_lambda,rho,eta] = tikhonov(U,s,V,b,lambda,x_0)
    [x_lambda,rho,eta] = tikhonov(U,sm,X,b,lambda,x_0) ,  sm = [sigma,mu]
    
    Computes the Tikhonov regularized solution x_lambda, given the SVD or
    GSVD as computed via csvd or cgsvd, respectively.  If the SVD is used,
    i.e. if U, s, and V are specified, then standard-form regularization
    is applied:
        min { || A x - b ||^2 + lambda^2 || x - x_0 ||^2 } .
    If, on the other hand, the GSVD is used, i.e. if U, sm, and X are
    specified, then general-form regularization is applied:
        min { || A x - b ||^2 + lambda^2 || L (x - x_0) ||^2 } .
    
    If an initial estimate x_0 is not specified, then x_0 = 0 is used.
    
    Note that x_0 cannot be used if A is underdetermined and L ~= I.
    
    If lambda is a vector, then x_lambda is a matrix such that
       x_lambda = [ x_lambda(1), x_lambda(2), ... ] .
    
    The solution norm (standard-form case) or seminorm (general-form
    case) and the residual norm are returned in eta and rho.
    Per Christian Hansen, DTU Compute, April 14, 2003.
    Reference: A. N. Tikhonov & V. Y. Arsenin, "Solutions of Ill-Posed
    Problems", Wiley, 1977.
    
    Modified by F. Parra-Rojas
    '''
    
    # Initialization.
    import sys
    import numpy as np
    
    if xlambda < 0:
        print('Illegal regularization parameter lambda')
        sys.exit
    
    m,_ = np.shape(U)
    n,_ = np.shape(V)


    if s.shape[1] == 1:    
        p = s.shape[0]
        beta = U[:,:p].T @ b
        ps = 1
        zeta = np.multiply(s,beta)
    else:
        p, ps = np.shape(s)
        beta = U[:,:p].T @ b
        s0 = np.reshape(s[:,0],(-1,1))
        s1 = np.reshape(s[:,1],(-1,1))
        
        zeta = np.multiply(s0,beta)

    ll = 1
    x_lambda = np.zeros((n,ll))
    rho = np.zeros((ll,1))
    eta = np.zeros((ll,1))
    
    
    # Treat each lambda separately.
    if ps == 1:
        # The standard-form case.
        x_lambda = V[:,:p] @ (np.divide(zeta,(s**2 + xlambda**2)))
        rho = (xlambda**2) * np.linalg.norm(np.divide(beta,(s**2 + xlambda**2)))
        eta = np.linalg.norm(x_lambda[:])
        if m > p:
            rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:,:n] @ np.concatenate([beta,U[:,p+1:n].T @ b]))**2)
        
    
    elif m >= n:
        # The overdetermined or square general-form case.
        gamma2 = (s0/s1)**2
    
        if p == n:
            x0 = np.zeros((n,1))
        else:
            x0 = V[:,p+1:n] @ U[:,p+1:n].T @ b

        xi = np.divide(zeta,(s0**2 + xlambda**2 * s1**2))

        x_lambda = V[:,:p] @ xi + x0
        
        rho = xlambda**2 * np.linalg.norm(beta/(gamma2 + xlambda**2))
        eta = np.linalg.norm(s1 * xi)
    
        if m > p:
            rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:,:n] @ np.concatenate([beta,U[:,p+1:n].T @ b]))**2)
    
    else:
        # The underdetermined general-form case.
        gamma2 = (s0/s1)**2
        if p == m:
            x0 = np.zeros((n,1))
        else:
            x0 = V[:,p+1:m] @ U[:,p+1:m].T @ b
        

        xi = zeta/(s0**2 + xlambda**2 * s1**2)
        x_lambda = V[:,:p] @ xi + x0
        rho = xlambda**2 * np.linalg.norm(beta/(gamma2 + xlambda**2))
        eta = np.linalg.norm(s1 * xi)
    
        if m > p:
            rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:,:n] @ np.concatenate([beta,U[:,p+1:n].T @ b]))**2)
    
    return x_lambda, rho, eta
    