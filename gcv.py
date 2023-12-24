#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:59:47 2020

@author: fran parra-rojas
"""

def gcv( U, s, b, method ):
    
    ''' 
    GCV Plot the GCV function and find its minimum.
    
    [reg_min,G,reg_param] = gcv(U,s,b,method)
    [reg_min,G,reg_param] = gcv(U,sm,b,method)  ,  sm = [sigma,mu]
    
    Plots the GCV-function
             || A*x - b ||^2
         G = -------------------
           (trace(I - A*A_I)^2
        as a function of the regularization parameter reg_param. Here, A_I is a
    matrix which produces the regularized solution.
        
    The following methods are allowed:
        method = 'Tikh' : Tikhonov regularization   (solid line )
        method = 'tsvd' : truncated SVD or GSVD     (o markers  )
        method = 'dsvd' : damped SVD or GSVD        (dotted line)
    If method is not specified, 'Tikh' is default.  U and s, or U and sm,
    must be computed by the functions csvd and cgsvd, respectively.
    
    If any output arguments are specified, then the minimum of G is
    identified and the corresponding reg. parameter reg_min is returned.
    Per Christian Hansen, DTU Compute, Dec. 16, 2003.
    Reference: G. Wahba, "Spline Models for Observational Data",
    SIAM, 1990.
    
    Adapted by Francisco Parra-Rojas
    '''
    
    import numpy as np
    import scipy.optimize as spo
    import matplotlib.pyplot as pt
    
    # Set defaults    
    npoints = 200 # Number of points on the curve.
    smin_ratio = 16*np.finfo(float).eps # Smallest regularization parameter.
    
    # Initiation
    m, n = np.shape(U)
    
    if s.shape[1] == 1:    
        p = s.shape[0]
        ps = 1
    else:
        p, ps = np.shape(s)
        s0 = np.reshape(s[:,0],(-1,1))
        s1 = np.reshape(s[:,1],(-1,1))
    

    beta = U.T @ b
    
    beta2 = np.linalg.norm(b)**2 - np.linalg.norm(beta)**2
    
    if ps == 2:
        s = np.flip(s0/s1)
        beta = np.flip(beta[:p])
    
    find_min = 1
    
    if method == 'Tikh' or method == 'tikh': 
        # Vector of regularization parameters.
        reg_param = np.zeros((npoints,1))
        G = np.zeros((npoints,1))
        s2 = s**2
        reg_param[npoints-1] = np.max([s[p-1],s[0]*smin_ratio])
        radio = (s[0]/reg_param[npoints-1])**(1/(npoints-1))
    
        for i in range(npoints-2,-1,-1):
            reg_param[i] = radio * reg_param[i+1]
        
        # Intrinsic residual.
        delta0 = 0

    
        if m > n and beta2 > 2:
            delta0 = beta2
    
        # Vector of GCV-function values.

        for i in range(0,npoints):
            #f = (reg_param[i]**2)/(s2 + reg_param[i]**2)
            #G[i] = (np.linalg.norm(f * beta[:p].T)**2 + delta0)/((m-n) + np.sum(f))**2
            G[i] = gcvfun( reg_param[i], s2, beta[:p], delta0, m-n, 0)
        # Plot GCV function.
        pt.figure(2)
        pt.loglog(reg_param,G,'-')
        pt.xlabel('$\lambda$')
        pt.ylabel('G($\lambda$)')
        pt.title('GCV function')
    
    # % Find minimum, if requested.
        if find_min:
            minG = np.min(G)
            minGi = np.argmin(G)

            #minGi = int(np.where(G == G.min())[0]) # Initial guess.
            # el valor real en Hansen es minGi+1 pero así da error
            x1 = reg_param[np.min([minGi,npoints])]
            
            x2 = reg_param[np.max([minGi-1,1])]

            if x2 < x1:
                reg_min = spo.fminbound(gcvfun,x2,x1,args=(s2,beta[:p],delta0,m-n,0),disp=0) # Minimizer.
            elif x1 < x2:
                reg_min = spo.fminbound(gcvfun,x1,x2,args=(s2,beta[:p],delta0,m-n,0),disp=0) # Minimizer. 
            elif x1 == x2:
                reg_min = x1
            #f1 = (reg_min**2)/(s2 + reg_min**2)
    
            minG=gcvfun( reg_min, s2, beta, delta0, m-n, 0 )

            pt.figure(2)
            pt.loglog(reg_min,minG,'*r',[reg_min,reg_min],[minG/1000,minG],':r')
            pt.title('GCV function, minimum at $\lambda$ = %1.5f' %reg_min)
    
    elif method == 'tsvd' or method == 'tgsvd':
        
        rho2 = np.zeros((p-1,1))
        rho2[p-2] = np.abs(beta[p-1])**2
        
        if m > n and beta2 > 0:
            rho2[p-2] = rho2[p-2] + beta2
        
        for k in range(p-3,-1,-1):
            rho2[k] = rho2[k+1] + np.abs(beta[k+1])**2
        
        G = np.zeros((p-1,1))
        
        for k in range(0,p-1):
            G[k] = rho2[k]/(m - k + (n-p))**2
        
        reg_param = np.arange(1,p)
        
        pt.figure(2)
        pt.semilogy(reg_param,G,'o')
        pt.xlabel('k')
        pt.ylabel('G(k)')
        pt.title('GCV function')
        
        if find_min:
            minG = np.min(G)
            reg_min = np.argmin(G)
            #reg_min = int(np.where(G == G.min())[0]) # Initial guess.
            pt.figure(2)
            pt.semilogy(reg_min, minG, '*r', [reg_min,reg_min],[minG/1000,minG],':r')
            pt.title('GCV function, minimum at k = %1.0i' %reg_min)
    
    elif method == 'dsvd':
        # Vector of regularization parameters.
        reg_param = np.zeros((npoints,1))
        G = np.zeros((npoints,1))
        reg_param[npoints-1] = np.max([s[p-1],s[0]*smin_ratio])

        radio = (s[0]/reg_param[npoints-1])**(1/(npoints-1))
        
        for i in range(npoints-2,-1,-1):
            reg_param[i] = radio * reg_param[i+1]
        
        # Intrinsic residual.
        delta0 = 0
    
        if m > n and beta2 > 2:
            delta0 = beta2
        # Vector of GCV-function values.
        for i in range(0,npoints):
            #f = (reg_param[i])/(s + reg_param[i])
            #G[i] = (np.linalg.norm(f * beta[:p])**2 + delta0)/((m-n) + np.sum(f))**2
            G[i] = gcvfun( reg_param[i], s, beta[:p], delta0, m-n, 1 )
    
        # Plot GCV function.
        pt.figure(2)
        pt.loglog(reg_param,G,':')
        pt.xlabel('$\lambda$')
        pt.ylabel('G($\lambda$)')
        pt.title('GCV function')
    
    # % Find minimum, if requested.
        if find_min:
            minG = np.min(G)
            minGi = np.argmin(G)
            #minGi = int(np.where(G == G.min())[0]) # Initial guess.
            # el valor real en Hansen es minGi+1 pero así da error
            x1 = reg_param[np.min([minGi,npoints])]
        
            x2 = reg_param[np.max([minGi-1,1])]
            if x2 < x1:
                reg_min = spo.fminbound(gcvfun,x2,x1,args=(s,beta[:p],delta0,m-n,1),disp=0) # Minimizer.
            elif x1 < x2:
                reg_min = spo.fminbound(gcvfun,x1,x2,args=(s,beta[:p],delta0,m-n,1),disp=0) # Minimizer.
            elif x1 == x2:
                reg_min = x1
        
            #f1 = (reg_min**2)/(s + reg_min**2)
    
            #minG = (np.linalg.norm(np.multiply(f1,beta[:p]))**2 + delta0)/((m-n) + np.sum(f1))**2 # Minimum of GCV function.
        
            
            minG = gcvfun( reg_min, s, beta[:p], delta0, m-n, 1 )
            pt.figure(2)
            pt.loglog(reg_min,minG,'*r',[reg_min,reg_min],[minG/1000,minG],':r')
            pt.title('GCV function, minimum at $\lambda$ = %1.5f' %reg_min)
        
        
    elif method == 'mtsv' or method == 'ttls':
        print('The MTSVD and TTLS methods are not supported')
        sys.exit()
    
    
    return reg_min, G, reg_param


def gcvfun( xlambda, s2, beta, delta0, mn, dsvd ):
    
    '''
    Auxiliary routine for gcv.  PCH, IMM, Feb. 24, 2008.
    
    Note: f = 1 - filter-factors.
    '''
    
    import numpy as np
    
    if not dsvd:
        f = (xlambda**2)/(s2 + xlambda**2)
    else: 
        f = (xlambda)/(s2 + xlambda)
        
    G = (np.linalg.norm(np.multiply(f,beta))**2 + delta0)/(mn + np.sum(f))**2
    return G