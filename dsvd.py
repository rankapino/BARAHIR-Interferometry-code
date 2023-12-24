#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:01:00 2020

@author: fran
"""

def dsvd( U, s, V, b, xlambda ):
    
    '''
    %DSVD Damped SVD and GSVD regularization.
    
    [x_lambda,rho,eta] = dsvd(U,s,V,b,lambda)
    [x_lambda,rho,eta] = dsvd(U,sm,X,b,lambda) ,  sm = [sigma,mu]
    
    Computes the damped SVD solution defined as
       x_lambda = V*inv(diag(s + lambda))*U'*b .
    If lambda is a vector, then x_lambda is a matrix such that
       x_lambda = [ x_lambda(1), x_lambda(2), ... ] .
    U, s, and V must be computed by the csvd function.
    
    If sm and X are specified, then the damped GSVD solution is computed:
    x_lambda = X*[ inv(diag(sigma + lambda*mu)) 0 ]*U'*b
                    [            0                 I ]
    U, sm, and X must be computed by the cgsvd function.
    
    The solution norm (standard-form case) or seminorm (general-form
    case) and the residual norm are returned in eta and rho.
    Reference: M. P. Ekstrom & R. L. Rhoads, "On the application of
    eigenvector expansions to numerical deconvolution", J. Comp.
    Phys. 14 (1974), 319-340.
    The extension to GSVD is by P. C. Hansen.
    Per Christian Hansen, DTU Compute, April 14, 2003.
    
    Adapted by Francisco Parra-Rojas
    '''
    
    import sys
    import numpy as np
    
    # Initialization.
    if type(xlambda) == int or type(xlambda) == float:
        ll = 1
        if xlambda < 0:
            print('Illegal regulariation parameter lambda')
            sys.exit()
        
        m = np.shape(U[:,0])[0]
        n = np.shape(V[0,:])[0]
    
        if len(s.shape) == 1:    
            p, = np.shape(s)
            beta = U[:,:p].T @ b
            ps = 1

        else:
            p, ps = np.shape(s)
            beta = U[:,:p].T @ b


        x_lambda = np.zeros((n,ll))
        
        # Treat each lambda separately.
        if ps == 1:
            # The standard-form case.
            x_lambda = (V[:,:p] @ (beta/(s + xlambda))).T
            rho = xlambda * np.linalg.norm(beta/(s + xlambda))
            eta = np.linalg.norm(x_lambda[:])
        
            if m > p:
                rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:,:n] @ np.concatenate([beta,U[:,p+1:n].T @ b]))**2)
    
        elif m >= n:
            # The overdetermined or square general-form case.
            x0 = V[:,p+1:n] @ U[:,p+1:n].T @ b
            xi = beta/(s[:,0] + xlambda*s[:,1])
            x_lambda[:] = V[:,:p] @ xi + x0
            rho = xlambda * np.linalg.norm(beta/(s[:,0]/s[:,1] + xlambda))
            eta = np.linalg.norm(s[:,1] * xi)
        
            if m > p:
                rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:,:n] @ np.concatenate([beta,U[:,p+1:n].T @ b]))**2)
    
        else:
            # The underdetermined general-form case.
            x0 = V[:,p+1:m] @ U[:,p+1:m].T @ b
            xi = beta/(s[:,0] + xlambda*s[:,1])
            x_lambda[:] = V[:,:p] @ xi + x0
            rho = xlambda * np.linalg.norm(beta/(s[:,0]/s[:,1] + xlambda))
            eta = np.linalg.norm(s[:,1] * xi)
        
    else:
        ll = len(xlambda)
        if min(xlambda) < 0:
            print('Illegal regulariation parameter lambda')
            sys.exit()
        
    
        m = np.shape(U[:,0])[0]
        n = np.shape(V[0,:])[0]
    
        if len(s.shape) == 1:    
            p, = np.shape(s)
            beta = U[:,:p].T @ b
            ps = 1

        else:
            p, ps = np.shape(s)
            beta = U[:,:p].T @ b


        x_lambda = np.zeros((n,ll))
        rho = np.zeros((ll,1))
        eta = np.zeros((ll,1))
    
        # Treat each lambda separately.
        if ps == 1:
            # The standard-form case.
            for i in range(0,ll):
                x_lambda[:,i] = V[:,:p] @ (beta/(s + xlambda[i]))
                rho[i] = xlambda[i] * np.linalg.norm(beta/(s + xlambda[i]))
                eta[i] = np.linalg.norm(x_lambda[:,i])
        
            if m > p:
                rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:,:n] @ np.concatenate([beta,U[:,p+1:n].T @ b]))**2)
    
        elif m >= n:
            # The overdetermined or square general-form case.
            x0 = V[:,p+1:n] @ U[:,p+1:n].T @ b
            for i in range(0,ll):
                xi = beta/(s[:,0] + xlambda[i]*s[:,1])
                x_lambda[:,i] = V[:,:p] @ xi + x0
                rho[i] = xlambda[i] * np.linalg.norm(beta/(s[:,0]/s[:,1] + xlambda[i]))
                eta[i] = np.linalg.norm(s[:,1] * xi)
        
            if m > p:
                rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:,:n] @ np.concatenate([beta,U[:,p+1:n].T @ b]))**2)
    
        else:
            # The underdetermined general-form case.
            x0 = V[:,p+1:m] @ U[:,p+1:m].T @ b
            for i in range(0,ll):
                xi = beta/(s[:,0] + xlambda[i]*s[:,1])
                x_lambda[:,i] = V[:,:p] @ xi + x0
                rho[i] = xlambda[i] * np.linalg.norm(beta/(s[:,0]/s[:,1] + xlambda[i]))
                eta[i] = np.linalg.norm(s[:,1] * xi)
    
    return x_lambda, rho, eta