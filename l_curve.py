#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:31:55 2020

@author: fran parra-rojas
"""

def l_curve( U, sm, b, method, L, V ):
    
    '''
    L_CURVE Plot the L-curve and find its "corner".
    
    [reg_corner,rho,eta,reg_param] =
                     l_curve(U,s,b,method)
                     l_curve(U,sm,b,method)  ,  sm = [sigma,mu]
                     l_curve(U,s,b,method,L,V)
    
    Plots the L-shaped curve of eta, the solution norm || x || or
    semi-norm || L x ||, as a function of rho, the residual norm
    || A x - b ||, for the following methods:
       method = 'Tikh'  : Tikhonov regularization   (solid line )
       method = 'tsvd'  : truncated SVD or GSVD     (o markers  )
       method = 'dsvd'  : damped SVD or GSVD        (dotted line)
       method = 'mtsvd' : modified TSVD             (x markers  )
    The corresponding reg. parameters are returned in reg_param.  If no
    method is specified then 'Tikh' is default.  For other methods use plot_lc.
    
    Note that 'Tikh', 'tsvd' and 'dsvd' require either U and s (standard-
    form regularization) computed by the function csvd, or U and sm (general-
    form regularization) computed by the function cgsvd, while 'mtvsd'
    requires U and s as well as L and V computed by the function csvd.
    
    If any output arguments are specified, then the corner of the L-curve
    is identified and the corresponding reg. parameter reg_corner is
    returned.  Use routine l_corner if an upper bound on eta is required.
    Reference: P. C. Hansen & D. P. O'Leary, "The use of the L-curve in
    the regularization of discrete ill-posed problems",  SIAM J. Sci.
    Comput. 14 (1993), pp. 1487-1503.
    Per Christian Hansen, DTU Compute, October 27, 2010.
    
    Adapted by Francisco Parra-Rojas
    '''
    
    import sys
    import numpy as np
    import matplotlib.pyplot as pt
    
    # Set defaults.
    npoints = 200 # Number of points on the L-curve for Tikh and dsvd.
    smin_ratio = 16*np.finfo(float).eps # Smallest regularization parameter.
    
    
    # Initialization.
    m, n = np.shape(U)
    
    if sm.shape[1] == 1:    
        p = sm.shape[0]
        ps = 1
    else:
        p, ps = np.shape(sm)
        s0 = np.reshape(sm[:,0],(-1,1))
        s1 = np.reshape(sm[:,1],(-1,1))

    locate = 1
    
    beta = U.T @ b
    
    beta2 = np.linalg.norm(b)**2 - np.linalg.norm(beta)**2
    
    if ps == 1:
        s = sm
        beta = beta[:p]
    else:
        s = np.flip(s0/s1)
        beta = np.flip(beta[:p])
    
    xi = beta[:p]/s
    
    xi[np.isinf(xi)] = 0
    
    if method == 'Tikh' or method == 'tikh':
    
        eta = np.zeros((npoints,1))
        rho = np.zeros((npoints,1))
        reg_param = np.zeros((npoints,1))
        s2 = s**2
    
        reg_param[npoints-1] = np.max([s[p-1],s[0]*smin_ratio])

        radio = (s[0]/reg_param[npoints-1])**(1/(npoints-1))
    
        for i in range(npoints-2,-1,-1):
            reg_param[i] = radio * reg_param[i+1]
    

        for j in range(0,npoints):
            f = s2/(s2 + reg_param[j]**2)
            eta[j] = np.linalg.norm(np.multiply(f,xi))
            rho[j] = np.linalg.norm(np.multiply((1-f),beta[:p]))
    
    
        if m > n and beta2 > 0:
            rho = np.sqrt(rho**2 + beta2)
    
        marker = '-'
        txt = 'Tikhonov'
    
    elif method == 'tsvd' or method == 'tgsvd':
        print('Not yet implemented')
        sys.exit()
        
#        eta = np.zeros((p,1))
#        rho = np.zeros((p,1))
#        
#        eta[0] = np.abs(xi[0])**2
#        
#        for k in range(1,p):
#            eta[k] = eta[k-1] + np.abs(xi[k])**2
#        
#        eta = np.sqrt(eta)
#        
#        if m > n:
#            if beta2 > 0:
#                rho[p-1] = beta2
#            else:
#                rho[p-1] = (np.finfo(float).eps)**2
#        
#        for k in range(p-1,-1,-1):
#            rho[k] = rho[k+1] + np.abs(beta[k+1])**2
#        
#        rho = np.sqrt(rho)
#        reg_param = np.arange(1,p+1)
#        marker = 'o'
#        
#        if ps == 1:
#            U = U[:,:p]
#            txt = 'TSVD'
#        else:
#            U = U[:,:p]
#            txt = 'TGSVD'
    
    elif method == 'mtsvd':
        print('Not yet implemented')
        sys.exit()
    
    elif method == 'dsvd':
        eta = np.zeros((npoints,1))
        rho = np.zeros((npoints,1))
        reg_param = np.zeros((npoints,1))
        reg_param[npoints-1] = np.max([s[p-1],s[0]*smin_ratio])
        
        radio = (s[0]/reg_param[npoints-1])**(1/(npoints-1))
    
        for i in range(npoints-2,-1,-1):
            reg_param[i] = radio * reg_param[i+1]
    

        for j in range(0,npoints):
            f = s/(s + reg_param[j])
            eta[j] = np.linalg.norm(np.multiply(f,xi))
            rho[j] = np.linalg.norm(np.multiply((1-f),beta[:p]))
    
    
        if m > n and beta2 > 0:
            rho = np.sqrt(rho**2 + beta2)
    
        marker = '-'
        if ps == 1:
            txt = 'DSVD'
        else:
            txt = 'DGSVD'
        
    else:
        print('Illegal method')
        sys.exit()
    

    # Locate the "corner" of the L-curve, if required.
    if locate:
        import l_corner as LCO
        reg_corner, rho_c, eta_c = LCO.l_corner( rho, eta, reg_param, U, sm, b, method) 
    
    # Make plot.
    import plot_lc as PTLC
    PTLC.plot_lc(rho,eta,marker,ps,reg_param)

    if locate:
        
        pt.loglog([np.min(rho)/100, rho_c],[eta_c,eta_c],':r',[rho_c,rho_c],[np.min(eta)/100, eta_c],':r')
        pt.title('L-curve %s corner at %1.3f' %(method, reg_corner))
        pt.show()
    
    return reg_corner, rho, eta, reg_param