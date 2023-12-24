#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:59:15 2020

@author: fran parra-rojas
"""

def mtsvd( U, s, V, b, k, L ):
    
    '''
    MTSVD Modified truncated SVD regularization.
    
    [x_k,rho,eta] = mtsvd(U,s,V,b,k,L)
    
    Computes the modified TSVD solution:
    x_k = V*[ xi_k ] .
               [ xi_0 ]
    Here, xi_k defines the usual TSVD solution
       xi_k = inv(diag(s(1:k)))*U(:,1:k)'*b ,
    and xi_0 is chosen so as to minimize the seminorm || L x_k ||.
    This leads to choosing xi_0 as follows:
        xi_0 = -pinv(L*V(:,k+1:n))*L*V(:,1:k)*xi_k .
    U, s, and V must be computed by the csvd function.
    
    The truncation parameter must satisfy k > n-p.
    
    If k is a vector, then x_k is a matrix such that
        x_k = [ x_k(1), x_k(2), ... ] .
    
    The solution and residual norms are returned in eta and rho.
    Reference: P. C. Hansen, T. Sekii & H. Shibahashi, "The modified
    truncated-SVD method for regularization in general form", SIAM J.
    Sci. Stat. Comput. 13 (1992), 1142-1150.
    Per Christian Hansen, IMM, 12/22/95.
    
    Adapted by Francisco Parra-Rojas
    '''
    
    # Initialization.
    import sys
    import numpy as np
    
    m,_ = np.shape(U)
    p,n = np.shape(L)
    
    if type(k) == int:
        lk = 1
        kmin = k
        kmax = k
    else:
        lk = len(k)
        kmin = int(min(k))
        kmax = int(max(k))
    
    
    if kmin < n-p+1 or kmax > n:
        print('Error: Illegal truncation parameter k')
        sys.exit()
    
    
    x_k = np.zeros((n,lk))
    
    rho = np.zeros((lk,1))
    eta = np.zeros((lk,1))
    
    
    beta = U[:,:n].T @ b
    
    xi = beta/s

    # Compute large enough QR factorization.    
    Q,R = np.linalg.qr(L @ V[:,-1:kmin-1:-1]);
    
    # Treat each k separately.
    if type(k) == int:
        kj = k
        xtsvd = V[:,:kj] @ xi[:kj]
        if kj == n:
            x_k[:] = xtsvd
        else:

            z = np.linalg.solve(R[:n-kj,:n-kj],Q[:,:n-kj].T @ (L @ xtsvd))
            z = z[n-kj::-1]
            xtsvd = xtsvd.reshape((len(xtsvd),1))
            x_k[:] = xtsvd - V[:,kj:n] @ z
        
        eta = np.linalg.norm(x_k[:])
        rho = np.linalg.norm(beta[kj:n] + np.multiply(s[kj:n],z))
    
    else:
        for j in range(0,lk):
            kj = int(k[j])
            xtsvd = V[:,:kj] @ xi[:kj]
            if kj == n:
                x_k[:,j] = xtsvd
            else:
                z = np.linalg.solve(R[:n-kj,:n-kj],Q[:,:n-kj].T @ (L @ xtsvd)) # estaba puesto (L @ xtsvd).T 
                z = z[n-kj::-1]
                xtsvd = xtsvd.reshape((len(xtsvd),1))

                x_k[:,j] = (xtsvd - V[:,kj:n] @ z).T
        
            eta[j] = np.linalg.norm(x_k[:,j])
            #rho[j] = np.linalg.norm(beta[kj:n] + s[kj:n] * z)
            rho[j] = np.linalg.norm(beta[kj:n] + np.multiply(s[kj:n],z))
    if m > n:
        rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:,:n] @ beta)**2)
        
    return x_k, rho, eta
        