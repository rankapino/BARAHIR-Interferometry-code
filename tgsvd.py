#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:15:23 2020

@author: fran parra-rojas
"""

def tgsvd( U, sm, X, b, k ):
    '''
    TGSVD Truncated GSVD regularization.
    
    [x_k,rho,eta] = tgsvd(U,sm,X,b,k) ,  sm = [sigma,mu]
    
    Computes the truncated GSVD solution
               [ 0              0                 0    ]
       x_k = X*[ 0  inv(diag(sigma(p-k+1:p)))     0    ]*U'*b .
               [ 0              0             eye(n-p) ]
    If k is a vector, then x_k is a matrix such that
       x_k = [ x_k(1), x_k(2), ... ] .
    U, sm, and X must be computed by the cgsvd function.
    
    The solution seminorm and the residual norm are returned in eta and rho.
    Reference: P. C. Hansen, "Regularization, GSVD and truncated GSVD",
    BIT 29 (1989), 491-504.
    Per Christian Hansen, DTU Compute, Feb. 24, 2008.
    
    Adapted by Francisco Parra-Rojas
    '''
    
    # Initialization
    import sys
    import numpy as np
    
    m,_ = np.shape(U)
    n,_ = np.shape(X)
    p,_ = np.shape(sm)
    
    
    if not np.shape(k):
    #if type(k) == int:
        
        lk = 1
    else:
        lk = len(k)
    
        if min(k) < 1 or max(k) > n:
            print('Error: Illegal truncation parameter k')
            sys.exit()

    
    x_k = np.zeros((n,lk))
    eta = np.zeros((lk,1)) 
    rho = np.zeros((lk,1))
    
    beta = U.T @ b
    
    xi = beta[:p]/sm[:,0]

    mxi = sm[:,1] * xi
    
    if m >= n:
        #The overdetermined or square case. Treat each k separately.
        if p == n:
            x_0 = np.zeros((n,1))
        else:
            if n-p == 1:
                x_0 = X[:,p:n] * (U[:,p:n].T @ b)
            elif n-p == 2:
                Ubnp = (U[:,p:n].T @ b)
                x_0 = X[:,p:n] @ Ubnp.reshape((len(Ubnp),1))
            

        if not np.shape(k):        
            i = k
            pi1 = p-i+1
            if i == 0:
                x_k = x_0
            else:
                ff = (X[:,pi1:p] @ xi[pi1:p].reshape((len(xi[pi1:p]),1)))
                x_k = (ff + x_0)

            
            rho = np.linalg.norm(beta[:p-i])
            eta = np.linalg.norm(mxi[pi1:p])
    
        else:
            for j in range(0,lk):
                i = int(k[j])
                pi1 = p-i+1
                if i == 0:
                    x_k[:,j] = x_0
                else:
                    
                    ff = (X[:,pi1:p] @ xi[pi1:p]).reshape((n,1))
                    x_k[:,j] = (ff + x_0).T
            
                rho[j] = np.linalg.norm(beta[:p-i])
                eta[j] = np.linalg.norm(mxi[pi1:p])
        
    if np.shape(U[:,0])[0] > n:
        rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:,:n] @ beta[:n])**2)
            
    return x_k, rho, eta
