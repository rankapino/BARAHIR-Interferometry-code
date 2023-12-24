#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:09:09 2020

@author: fran parra-rojas
"""

def tsvd( U, s, V, b, k ):
    
    '''
    TSVD Truncated SVD regularization.

    [x_k,rho,eta] = tsvd(U,s,V,b,k)

    Computes the truncated SVD solution
        x_k = V(:,1:k)*inv(diag(s(1:k)))*U(:,1:k)'*b .
    If k is a vector, then x_k is a matrix such that
        x_k = [ x_k(1), x_k(2), ... ] .
    U, s, and V must be computed by the csvd function.

    The solution and residual norms are returned in eta and rho.
    Per Christian Hansen, DTU Compute, 12/21/97.
    
    Adapted by Francisco Parra-Rojas
    '''


    #Initialization.
    
    import sys
    import numpy as np
    
    n = np.shape(V[:,0])[0]
    p = np.shape(V[0,:])[0]
    
    if type(k) == int or type(k) == np.int64 or type(k) == np.int32:
        lk = 1
    else:
        lk = len(k)
    
        if min(k) < 1 or max(k) > n:
            print('Error: Illegal truncation parameter k')
            sys.exit()
    
    x_k = np.zeros((n,lk))
    
    rho = np.zeros((lk,1))
    eta = np.zeros((lk,1))
    
    
    beta = U[:,:p].T @ b
    
    xi = beta/s
    
    # Treat each k separately
    if type(k) == int or type(k) == np.int64 or type(k) == np.int32:
        i = k
        x_k = V[:,:i] @ xi[:i]
        
        eta = np.linalg.norm(xi[:i])
        rho = np.linalg.norm(beta[i+1:p])
    
    else:
        for j in range(0,lk):
            i = int(k[j])
            ff = (V[:,:i] @ xi[:i])
            x_k[:,j] = ff.T
            eta[j] = np.linalg.norm(xi[:i])
            rho[j] = np.linalg.norm(beta[i+1:p])
    
    if np.shape(U[:,0])[0] > p:
        rho = np.sqrt(rho**2 + np.linalg.norm(U[:,:p] @ beta)**2)
    
    return x_k, eta, rho