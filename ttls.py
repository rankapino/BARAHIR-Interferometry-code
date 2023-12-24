#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:00:47 2020

@author: fran parra-rojas
"""

def ttls( V1, k, s1 ):
    
    '''
    TTLS Truncated TLS regularization.

    [x_k,rho,eta] = ttls(V1,k,s1)

    Computes the truncated TLS solution
        x_k = - V1(1:n,k+1:n+1)*pinv(V1(n+1,k+1:n+1))
    where V1 is the right singular matrix in the SVD of the matrix
        [A,b] = U1*diag(s1)*V1' .
    V1 and s1 must be computed via [U1,s1,V1] = csvd([A,b],'full').

    If k is a vector, then x_k is a matrix such that
        x_k = [ x_k(1), x_k(2), ... ] .
    If k is not specified, k = n is used.

    The solution norms and TLS residual norms corresponding to x_k are
    returned in eta and rho, respectively.  Notice that the singular
    values s1 are required to compute rho.
    Reference: R. D. Fierro, G. H. Golub, P. C. Hansen and D. P. O'Leary,
    "Regularization by truncated total least squares", SIAM J. Sci. Comput.
    18 (1997), 1223-1241.
    Per Christian Hansen, DTU Compute, 03/18/93.
    
    Adapted by Francisco Parra-Rojas
    '''

    #Initialization.
    
    import sys
    import numpy as np
    
    n1 = np.shape(V1[:,0])[0]
    m1 = np.shape(V1[0,:])[0]
     
    n = n1-1
     
    if m1 != n1:
        print('Error: The matrix V1 must be square')
        sys.exit()
    
    if type(k) == int:
        lk = 1
    else:
        lk = len(k)
    
        if min(k) < 1 or max(k) > n:
            print('Error: Illegal truncation parameter k')
            sys.exit()
    
    x_k = np.zeros((n,lk))
    
    ns = len(s1)
    rho = np.zeros((lk,1))
    eta = np.zeros((lk,1))

    # Treat each k separately
    if type(k) == int:
        i = k
        v = V1[n,i+1:n1]
        gamma = 1/(v @ v.T)
        x_k = -V1[0:n,i+1:n1] @ v.T * gamma
        rho = np.linalg.norm(s1[i+1:ns])
        eta = np.sqrt(gamma-1)
    
    else:
        for j in range(0,lk):
            i = int(k[j])
            v = V1[n,i+1:n1]
            gamma = 1/(v @ v.T)
            x_k[:,j] = -V1[0:n,i+1:n1] @ v.T * gamma
            rho[j] = np.linalg.norm(s1[i+1:ns])
            eta[j] = np.sqrt(gamma-1)

        
    return x_k, rho, eta