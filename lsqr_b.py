#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:03:31 2020

@author: fran parra-rojas
"""

def lsqr_b( A, b, k, reorth, s ):
    
    '''
    LSQR_B Solution of least squares problems by Lanczos bidiagonalization.
    
    [X,rho,eta,F] = lsqr_b(A,b,k,reorth,s)
    
    Performs k steps of the LSQR Lanczos bidiagonalization algorithm
    applied to the system
       min || A x - b || .
    The routine returns all k solutions, stored as columns of
    the matrix X.  The solution norm and residual norm are returned
    in eta and rho, respectively.
    
    If the singular values s are also provided, lsqr computes the
    filter factors associated with each step and stores them columnwise
    in the matrix F.
    
    Reorthogonalization is controlled by means of reorth:
       reorth = 0 : no reorthogonalization (default),
       reorth = 1 : reorthogonalization by means of MGS.
    Reference: C. C. Paige & M. A. Saunders, "LSQR: an algorithm for
    sparse linear equations and sparse least squares", ACM Trans.
    Math. Software 8 (1982), 43-71.
    Per Christian Hansen, IMM, August 13, 2001.
    
    Adapted by Francisco Parra-rojas
    '''
    
    import sys
    import numpy as np
    
    # The fudge threshold is used to prevent filter factors from exploding.
    fudge_thr = 1e-4
    
    # Initialization.
    if k < 1:
        print('Number of steps k must be positive')
        sys.exit()
        
    m,n = np.shape(A)
    X = np.zeros((n,k))
    
    if reorth == 0:
        UV = 0
    elif reorth == 1:
        U = np.zeros((m,k+1))
        V = np.zeros((n,k+1))
        UV = 1
        if k >= n:
            print('No. of iterations must satisfy k < n')
            sys.exit()
    else:
        print('Illegal reorth')
        sys.exit()
    
    eta = np.zeros((k,1))
    rho = np.zeros((k,1))
    
    c2 = -1
    s2 = 0
    xnorm = 0
    z = 0
    
    ls = len(s)
    F = np.zeros((ls,k))
    Fv = np.zeros((ls,1))
    Fw = np.zeros((ls,1))
    s = s**2
    
    # Prepare for LSQR iteration.
    v = np.zeros((n,1))
    x = np.zeros((n,1))
    beta = np.linalg.norm(b)
    
    if beta == 0:
        print('Right-hand side must be nonzero')
        sys.exit()
        
    u = b/beta
    
    if UV:
        U[:,0] = u
        
    r = A.T @ u
    alpha = np.linalg.norm(r)
    v = r/alpha
    
    if UV:
        V[:,0]=v
    
    phi_bar = beta
    rho_bar = alpha
    w = v
    
    Fv = s/(alpha * beta)
    Fw = Fv

    # Perform Lanczos bidiagonalization with/without reorthogonalization.
    for i in range(1,k+1):
        alpha_old = alpha
        beta_old = beta
        
        # Compute A*v - alpha*u.
        p = A @ v.T - alpha * u
        
        if reorth == 0:
            beta = np.linalg.norm(p)
            u = p/beta
        else:
            for j in range(0,i-1):
                p = p - (U[:,j] @ p) * U[:,j]
            
            beta = np.linalg.norm(p)
            u = p/beta
            
        # Compute A'*u - beta*v.
        r = A.T @ u - beta * v
        
        if reorth == 0:
            alpha = np.linalg.norm(r)
            v = r/alpha
        else:
            for j in range(0,i-1):
                r = r - (V[:,j] @ r) * V[:,j]
                
            alpha = np.linalg.norm(r)
            v = r/alpha
        # Store U and V if necessary.
        if UV:
            U[:,i] = u
            V[:,i] = v
        
        # Construct and apply orthogonal transformation.
        rrho = np.linalg.norm([rho_bar,beta])
        c1 = rho_bar/rrho
        s1 = beta/rrho

        theta = s1 * alpha
        rho_bar = -c1 * alpha
        
        phi = c1 * phi_bar
        phi_bar = s1 * phi_bar
        
        # Compute solution norm and residual norm if necessary;
        delta = s2 * rrho
        gamma_bar = -c2 * rrho
        rhs = phi - delta * z
        
        z_bar = rhs/gamma_bar
        eta[i-1] = np.linalg.norm([xnorm,z_bar])
        
        gamma = np.linalg.norm([gamma_bar,theta])
        
        c2 = gamma_bar/gamma
        s2 = theta/gamma
        
        z = rhs/gamma
        xnorm = np.linalg.norm([xnorm,z])
        
        rho[i-1] = np.abs(phi_bar)
        
        # If required, compute the filter factors.
        if i == 1:
            Fv_old = Fv
            Fv = Fv * (s - beta**2 - alpha_old**2)/(alpha * beta)
            F[:,i-1] = (phi/rrho) * Fw
        else:
            tmp = Fv
            Fv = (Fv * (s - beta**2 - alpha_old**2) - Fv_old * alpha_old * beta_old)/(alpha * beta)
            FV_old = tmp
            F[:,i-1] = F[:,i-2] + (phi/rrho) * Fw
        
        if i > 2:
            
            f = np.argwhere((np.abs(F[:,i-2]-1) < fudge_thr) & (np.abs(F[:,i-3]-1) < fudge_thr))
            if not np.all(f==0):
                F[f,i-1] = np.ones((len(f),1))
        
        Fw = Fv - (theta/rrho) * Fw
        
        # Update the solution.
        x = x + (phi/rrho) * w
        w = v - (theta/rrho) * w
        
        X[:,i-1] = x[0]
        
    return X, rho, eta, F