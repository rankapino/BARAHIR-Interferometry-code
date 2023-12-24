#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:46:21 2020

@author: fran parra-rojas
"""

def lagrange( U, s, b ):
    
    '''
    LAGRANGE Plot the Lagrange function for Tikhonov regularization.
    
    [La,dLa,lambda0] = lagrange(U,s,b,more)
    [La,dLa,lambda0] = lagrange(U,sm,b,more)  ,  sm = [sigma,mu]
    
    Plots the Lagrange function
       La(lambda) = || A x - b ||^2 + lambda^2*|| L x ||^2
    and its first derivative dLa = dLa/dlambda versus lambda.
    Here, x is the Tikhonov regularized solution.  U and s, or U and sm,
    must be computed by the functions csvd and cgsvd, respectively.
    
    If nargin = 4, || A x - b || and || L x || are also plotted.
    
    Returns La, dLa, and the value lambda0 of lambda for which
    dLa has its minimum.
    Per Christian Hansen, DTU Compute, Feb. 21, 2001.
    
    Adapted by Francisco Parra-Rojas
    '''
    
    import numpy as np
    import matplotlib.pyplot as pt
    
    # Set default number of points.
    npoints = 200
    
    # Initialization.
    m,n = np.shape(U)
    
    if len(s.shape) == 1:    
        p, = np.shape(s)
        ps = 1
    else:
        p, ps = np.shape(s)
    
    
    beta = U.T @ b
    beta2 = np.linalg.norm(b)**2 - np.linalg.norm(beta)**2
    
    if ps == 2:
        
        s = np.flip(s[:,0]/s[:,1])
        beta = np.flip(beta)
    
    xi = beta[:p]/s
    
    # Compute the L-curve.
    eta = np.zeros((npoints,1))
    rho = np.zeros((npoints,1))
    xlambda = np.zeros((npoints,1))
    

    xlambda[npoints-1] = s[p-1]
    ratio = (s[0]/s[p-1])**(1/(npoints-1))
    
    for i in range(npoints-2,-1,-1):
        xlambda[i] = ratio * xlambda[i+1]
    
    import fil_fac as FF
    for i in range(0,npoints):
        f = FF.fil_fac(s,xlambda[i])
        eta[i] = np.linalg.norm(f * xi)
        rho[i] = np.linalg.norm((1-f)*beta[:p])
    
    if m > n and beta2 > 0:
        rho = np.sqrt(rho**2 + beta2)
    
    # Compute the Lagrange function and its derivative.
    La = rho**2 + (xlambda**2)*(eta**2)
    dLa = 2*xlambda * (eta**2)
    
    mindLa = np.min(dLa)
    mindLi = int(np.where(dLa == dLa.min())[0])
    
    lambda0 = xlambda[mindLi]
    
    # Plot the functions.
    pt.figure(1)
    pt.loglog(xlambda,La,'-',xlambda,dLa,'--',lambda0,mindLa,'o')
    pt.legend('La','dLa/d$\lambda$')
    
    pt.xlabel('$\lambda$')
    pt.title('Lagrange function La and its derivative')
    
    return La, dLa, lambda0
    