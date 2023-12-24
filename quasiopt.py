#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:34:21 2020

@author: fran parra-rojas
"""

def quasiopt( U, sm, b, method ):
    
    '''
    QUASIOPT Quasi-optimality criterion for choosing the reg. parameter
    
    [reg_min,Q,reg_param] = quasiopt(U,s,b,method)
    [reg_min,Q,reg_param] = quasiopt(U,sm,b,method)  ,  sm = [sigma,mu]
    
    Plots the quasi-optimality function Q for the following methods:
    method = 'Tikh' : Tikhonov regularization   (solid line )
       method = 'tsvd' : truncated SVD or GSVD     (o markers  )
       method = 'dsvd' : damped SVD or GSVD        (dotted line)
    If no method is specified, 'Tikh' is default.  U and s, or U and sm,
    must be computed by the functions csvd and cgsvd, respectively.
    
    If any output arguments are specified, then the minimum of Q is
    identified and the corresponding reg. parameter reg_min is returned.
    Per Christian Hansen, DTU Compute, Feb. 21, 2001.
    
    Adapted by Francisco Parra-Rojas
    '''
    
    import numpy as np
    import scipy.optimize as spo
    import matplotlib.pyplot as pt

    # Set defaults.
    npoints = 200 # Number of points for 'Tikh' and 'dsvd'.
    
    # Initialization.
    if len(sm.shape) == 1:
        p, = np.shape(sm)
        ps = 1
    else:
        p, ps = np.shape(sm)
    
    if ps == 1:
        s = sm
    else:
        s = np.flip(sm[:,0]/sm[:,1])
        U = np.flip(U[:,:p],0)
        s = np.reshape(s,(-1,1))

    
    xiii = np.dot(U.T,b)
    xi = np.divide(xiii,s)

    
    
    find_min = 1
    
    # Compute the quasioptimality function Q.
    if method == 'Tikh' or method == 'tikh':
        # Compute a vector of Q-values.
        Q = np.zeros((npoints,1))
        reg_param = np.zeros((npoints,1))
    
        reg_param[npoints-1] = s[p-1]
    
        radio = (s[0]/s[p-1])**(1/(npoints-1))
    
        for i in range(npoints-2,-1,-1):
            reg_param[i] = radio * reg_param[i+1]
        
        for i in range(0,npoints):
            Q[i] = quasifun(reg_param[i],s,xi,method)
            

    # Find the minimum, if requested.
        if find_min:
            minQ = np.min(Q)
            minQi = np.argmin(Q)
            #minQi = int(np.where(Q == Q.min())[0]) # Initial guess.
            #reg_min = reg_param[minQi-1]
            reg_min = 0.0
            x1 = reg_param[np.min([minQi+1,npoints-1])]
            x2 = reg_param[np.max([minQi-1,1])]
        
        # Minimizer.
            if x2 < x1:
                reg_min = spo.fminbound(quasifun,x2,x1,args=(s,xi,method),disp=0)
            elif x1 < x2:    
                reg_min = spo.fminbound(quasifun,x1,x2,args=(s,xi,method),disp=0)
            elif x1 == x2:
                reg_min = x1
        
            minQ = quasifun(reg_min,s,xi,method) # Minimum of function.
    
    elif method == 'dsvd':
        Q = np.zeros((npoints,1))
        reg_param = np.zeros((npoints,1))
    
        reg_param[npoints-1] = s[p-1]
    
        radio = (s[0]/s[p-1])**(1/(npoints-1))
    
        for i in range(npoints-2,-1,-1):
            reg_param[i] = radio * reg_param[i+1]
    
        for i in range(0,npoints):
            Q[i] = quasifun(reg_param[i],s,xi,method)
    
    # Find the minimum, if requested.
        if find_min:
            minQ = np.min(Q)
            minQi = np.argmin(Q)
            #minQi = int(np.where(Q == Q.min())[0]) # Initial guess.
            #OJO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            reg_min = 0.0
            
            x1 = reg_param[np.min([minQi+1,npoints-1])]
            x2 = reg_param[np.max([minQi-1,1])]
        
        # Minimizer.
            if x2 < x1:
                reg_min = spo.fminbound(quasifun,x2,x1,args=(s,xi,method),disp=0)
            elif x1 < x2:    
                reg_min = spo.fminbound(quasifun,x1,x2,args=(s,xi,method),disp=0)
            elif x1 == x2:
                reg_min = x1
        
            minQ = quasifun(reg_min,s,xi,method) # Minimum of function.
    
    elif method == 'tsvd' or method == 'tgsvd':
        Q = np.abs(xi)
        reg_param = np.arange(1,p+1)

        
        if find_min:
            minQ = np.min(Q)
            minQi = np.argmin(Q)
            #minQi = int(np.where(Q == Q.min())[0])
            reg_min = reg_param[minQi-1]
    
    else:
        print('Illegal Method')
        sys.exit()
    
    if method == 'tsvd' or method == 'tgsvd':
        
        pt.figure(2)
        pt.semilogy(reg_param,Q,'o')
        pt.xlabel('$\lambda$')
        pt.ylabel('Q($\lambda$)')
        pt.title('Quasi-optimality function')
        
        if find_min:
            pt.loglog([reg_min,reg_min],[minQ,minQ/1000],'--')
            pt.title('Quasi-optimality function, minimum at %1.5f' %reg_min )
        
    else:
        if method == 'tikh' or method == 'Tikh' or method == 'dsvd':
            pt.figure(2)
            pt.loglog(reg_param,Q)
            pt.xlabel('$\lambda$')
            pt.ylabel('Q($\lambda$)')
        else:
            pt.figure(2)
            pt.loglog(reg_param,Q,':')
            pt.xlabel('$\lambda$')
            pt.ylabel('Q($\lambda$)')
            
    
        if find_min:
            pt.loglog([reg_min,reg_min],[minQ,minQ/1000],'--')
            pt.title('Quasi-optimality function minimum at %1.5f' %reg_min )
        
    
    
    return reg_min, Q, reg_param
    
    
def quasifun(xlambda,s,xi,method):
    
    '''
    Auxiliary routine for quasiopt.  PCH, IMM, 12/29/97.
    
    Adapted by Francisco Parra-Rojas
    '''
    
    import numpy as np
    
    if method == 'dsvd':
        f = s/(s + xlambda)
    else:
        f = (s**2)/(s**2 + xlambda**2)
    
    Q1 = np.multiply(1-f,f)
    Q = np.linalg.norm(np.multiply(Q1,xi))
    
    return Q