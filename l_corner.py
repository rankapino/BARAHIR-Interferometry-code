#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:59:01 2020

@author: fran parra-rojas
"""

def l_corner( rho, eta, reg_param, U, s, b, method ):
    
    '''
    L_CORNER Locate the "corner" of the L-curve.
    
    [reg_c,rho_c,eta_c] =
           l_corner(rho,eta,reg_param)
           l_corner(rho,eta,reg_param,U,s,b,method,M)
           l_corner(rho,eta,reg_param,U,sm,b,method,M) ,  sm = [sigma,mu]
    
    Locates the "corner" of the L-curve in log-log scale.
    
    It is assumed that corresponding values of || A x - b ||, || L x ||,
    and the regularization parameter are stored in the arrays rho, eta,
    and reg_param, respectively (such as the output from routine l_curve).
    
    If nargin = 3, then no particular method is assumed, and if
    nargin = 2 then it is issumed that reg_param = 1:length(rho).
    
    If nargin >= 6, then the following methods are allowed:
       method = 'Tikh'  : Tikhonov regularization
       method = 'tsvd'  : truncated SVD or GSVD
       method = 'dsvd'  : damped SVD or GSVD
       method = 'mtsvd' : modified TSVD,
    and if no method is specified, 'Tikh' is default.  If the Spline Toolbox
    is not available, then only 'Tikh' and 'dsvd' can be used.
    
    An eighth argument M specifies an upper bound for eta, below which
    the corner should be found.
    Per Christian Hansen, DTU Compute, January 31, 2015. 
    
    Adapted by Francisco Parra-Rojas
    '''
    
    import sys
    import numpy as np
    import scipy.optimize as spo
    
    # Ensure that rho and eta are column vectors.
    rho = rho[:]
    eta = eta[:]
    
    # Set this logical variable to 1 (true) if the corner algorithm
    # should always be used, even if the Spline Toolbox is available.
    alwayscorner = 0
    
    # Set threshold for skipping very small singular values in the
    # analysis of a discrete L-curve.
    s_thr = np.finfo(float).eps # Neglect singular values less than s_thr.
    
    # Set default parameters for treatment of discrete L-curve.
    deg = 2 # Degree of local smooting polynomial.
    q = 2 # Half-width of local smoothing interval.
    order = 4 # Order of fitting 2-D spline curve.
    
    # Initialization.
    if len(rho) < order:
        print('ERROR: Too few data points for L-curve analysis')
        sys.exit()
    
    
    if s.shape[1] == 1:    
        p = s.shape[0]
        ps = 1
    else:
        p, ps = np.shape(s)
        s0 = np.reshape(s[:,0],(-1,1))
        s1 = np.reshape(s[:,1],(-1,1))
    
    m, n = np.shape(U)

    
    beta = U.T @ b
    b0 = b - U @ beta
    
    if ps == 2:
        s = np.flip(s0/s1)
        beta = np.flip(beta[:p])
    

    xi = beta/s
    
    # Take of the least-squares residual.
    if m > n:
        beta = np.append(beta,np.linalg.norm(b0))
    if method == 'Tikh' or method == 'tikh':
        # The L-curve is differentiable; computation of curvature in
        # log-log scale is easy.
  
        # Compute g = - curvature of L-curve.
        g = lcfun(reg_param,s,beta,xi,0)
    
        # Locate the corner.  If the curvature is negative everywhere,
        # then define the leftmost point of the L-curve as the corner.
        gi = np.argmin(g)

    
        # en eloriginal es gi+1 
        x1 = reg_param[np.min([gi,len(g)])]
    
        x2 = reg_param[np.max([gi-1,1])]
    
        reg_c = spo.fminbound(lcfun,x1,x2,args=(s,beta,xi,0),disp=0) # Minimizer.
    
        kappa_max = -lcfun(reg_c,s,beta,xi,0) # Maximum curvature.
    
        if kappa_max < 0:
            lr = len(rho)
            reg_c = reg_param[lr-1]
            rho_c = rho[lr-1]
            eta_c = eta[lr-1]
        else:
            f = (s**2)/(s**2 + reg_c**2)
            eta_c = np.linalg.norm(f * xi)
            rho_c = np.linalg.norm((1-f) * beta[:len(f)])
            if m > n:
                rho_c = np.sqrt(rho_c**2 + np.linalg.norm(b0)**2)
    
    elif method == 'dsvd':
        
        g = lcfun(reg_param,s,beta,xi,1)
        
        # Locate the corner.  If the curvature is negative everywhere,
        # then define the leftmost point of the L-curve as the corner.
        gi = np.argmin(g)
    
        # en eloriginal es gi+1 
        x1 = reg_param[np.min([gi,len(g)])]
    
        x2 = reg_param[np.max([gi-1,1])]
        
        if x1 < x2:
            reg_c = spo.fminbound(lcfun,x1,x2,args=(s,beta,xi,1),disp=0) # Minimizer.
        elif x1 > x2: 
            reg_c = spo.fminbound(lcfun,x2,x1,args=(s,beta,xi,1),disp=0)
        elif x1 == x2:
            reg_c = x1
            
        kappa_max = -lcfun(reg_c,s,beta,xi,1) # Maximum curvature.
        
        if kappa_max < 0:
            lr = len(rho)
            reg_c = reg_param[lr-1]
            rho_c = rho[lr-1]
            eta_c = eta[lr-1]
        else:
            f = (s)/(s + reg_c)
            eta_c = np.linalg.norm(f * xi)
            rho_c = np.linalg.norm((1-f) * beta[:len(f)])
            if m > n:
                rho_c = np.sqrt(rho_c**2 + np.linalg.norm(b0)**2)
                
    elif method == 'tsvd' or method == 'tgsvd' or method == 'mtsvd':
        print('Not yet implemented')
        sys.exit()
            
    else:
        print('Illegal method')
        sys.exit()
            
    return reg_c, rho_c, eta_c

def lcfun(xlambda,s,beta,xi,fifth):
    
    '''
    Auxiliary routine for l_corner; computes the NEGATIVE of the curvature.
    Note: lambda may be a vector.  PCH, DTU Compute, Jan. 31, 2015.
    '''
    
    # Initialization.
    import numpy as np
    
    phi = np.zeros(np.shape(xlambda))
    dphi = np.zeros(np.shape(xlambda))
    psi = np.zeros(np.shape(xlambda))
    dpsi = np.zeros(np.shape(xlambda))
    eta = np.zeros(np.shape(xlambda))
    rho = np.zeros(np.shape(xlambda))
    
    # A possible least squares residual.
    if len(beta) > len(s):
        LS = True
        rhoLS2 = beta[-1]**2
        beta = beta[:-1]
    else:
        LS = False
    
    # Compute some intermediate quantities.
    for i in range(len(xlambda)):
        
        if fifth:
            f = s/(s + xlambda[i])
        else:    
            f = (s**2)/(s**2 + xlambda[i]**2)
        cf = 1-f
        eta[i] = np.linalg.norm(np.multiply(f,xi))
        rho[i] = np.linalg.norm(np.multiply(cf,beta))
        f1 = -2 * f * cf/xlambda[i]
        f2 = -f1 * (3 - 4*f)/xlambda[i]
        phi[i] = np.sum(np.multiply(f,np.multiply(f1,np.power(np.abs(xi),2))))
        psi[i] = np.sum(np.multiply(cf,np.multiply(f1,np.power(np.abs(beta),2))))
        dphi[i] = np.sum(np.multiply((np.power(f1,2) + np.multiply(f,f2)),np.power(np.abs(xi),2)))
        dpsi[i] = np.sum(np.multiply((-np.power(f1,2) + np.multiply(cf,f2)),np.power(np.abs(beta),2)))
       
    # Take care of a possible least squares residual.
    if LS:
        rho = np.sqrt(rho**2 + rhoLS2)
    
    # Now compute the first and second derivatives of eta and rho
    # with respect to lambda;
    deta = phi/eta
    drho = -psi/rho
    ddeta = dphi/eta - deta*(deta/eta)
    ddrho = -dpsi/rho - drho*(drho/rho)
    
    # Convert to derivatives of log(eta) and log(rho).
    dlogeta = deta/eta
    dlogrho = drho/rho
    ddlogeta = ddeta/eta - dlogeta**2
    ddlogrho = ddrho/rho - dlogrho**2
    
    # Let g = curvature.
    g = -(dlogrho*ddlogeta - ddlogrho*dlogeta)/(dlogrho**2 + dlogeta**2)**1.5
    
    return g