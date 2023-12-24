#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:45:41 2020

@author: fran parra-rojas
"""

def ncp( U, sm, b, method ):
    
    '''
    NCP Plot the NCPs and find the one closest to a straight line.
    
    [reg_min,G,reg_param] = ncp(U,s,b,method)
    [reg_min,G,reg_param] = ncp(U,sm,b,method)  ,  sm = [sigma,mu]
    
    Plots the normalized cumulative priodograms (NCPs) for the residual
    vectors A*x - b.  The following methods are allowed:
       method = 'Tikh' : Tikhonov regularization
       method = 'tsvd' : truncated SVD or GSVD
       method = 'dsvd' : damped SVD or GSVD
    If method is not specified, 'Tikh' is default.  U and s, or U and sm,
    must be computed by the functions csvd and cgsvd, respectively.
    
    The NCP closest to a straight line is identified and the corresponding
    regularization parameter reg_min is returned.  Moreover, dist holds the
    distances to the straight line, and reg_param are the corresponding
    regularization parameters.
    Per Christian Hansen, DTU Compute, Jan. 4, 2008.
    Reference: P. C. Hansen, M. Kilmer & R. H. Kjeldsen, "Exploiting
    residual information in the parameter choice for discrete ill-posed
    problems", BIT 46 (2006), 41-59.
    
    Adapted by Francisco Parra-Rojas
    '''
    
    # Set defaults.
    import numpy as np
    import scipy.optimize as spo
    import matplotlib.pyplot as pt
    
    npoints = 200 # Number of initial NCPS for Tikhonov.
    nNCPs = 20 # Number of NCPs shown for Tikhonov.
    smin_ratio = 16*np.finfo(float).eps # Smallest regularization parameter.
    
    # Initialization.
    if sm.shape[1] == 1:    
        p = sm.shape[0]
        ps = 1
    else:
        p, ps = np.shape(sm)
        s0 = np.reshape(sm[:,0],(-1,1))
        s1 = np.reshape(sm[:,1],(-1,1))
    
    m,_ = np.shape(U)

    
    beta = U.T @ b
    
    if ps == 1:
        s = sm
    else:
        s = np.flip(s0/s1)
        beta = np.flip(beta[:p])
    
    if method == 'Tikh' or method == 'tikh':
        
        # Vector of regularization parameters.
        reg_param = np.zeros((npoints,1))
    
        reg_param[npoints-1] = np.max([s[p-1],s[0]*smin_ratio])

        radio = (s[0]/reg_param[npoints-1])**(1/(npoints-2))
    
        for i in range(npoints-2,-1,-1):
            reg_param[i] = radio * reg_param[i+1]
    
        # Vector of distances to straight line.
        dists = np.zeros((npoints,1))    
        if np.isreal(np.all(beta)):
            print('real')
            q = int(np.floor(m/2))
        else:
            print('no real')
            q = int(m-1)
    
        cp = np.zeros((q,npoints))
    
        #s_res = np.reshape(s,(len(s),1))
        #beta_res = np.reshape(beta[:p],(len(beta[:p]),1)) 
    
    
        for i in range(0,npoints):
            dists[i], cp[:,i] = ncpfun2(reg_param[i],s,beta,U[:,:p],0)

        # Plot selected NCPs.
        stp = int(np.around(npoints/nNCPs))
        
        pt.figure(2)
        pt.plot(cp[:,0:npoints:stp])

        # Find minimum.
        minG = np.min(dists)
        minGi = np.argmin(dists)
        #minGi = int(np.where(dists == dists.min())[0]) # Initial guess.
    
    
        x1 = reg_param[np.min([minGi+1,npoints-1])]
        x2 = reg_param[np.max([minGi-1,1])]
    
        # Minimizer.
        if x2 < x1:
            reg_min = spo.fminbound(ncpfun,x2,x1,args=(s,beta[:p],U[:,:p],0),disp=0) 
        elif x1 < x2:    
            reg_min = spo.fminbound(ncpfun,x1,x2,args=(s,beta,U[:,:p],0),disp=0)
        elif x1 == x2:
            reg_min = x1

        dist,cp = ncpfun3(reg_min,s,beta[:p],U[:,:p],0)
#    
        pt.plot(cp,'-r', linewidth=2)
        pt.title('Selected NCPs. Most white for $\lambda$ = %1.5f' %reg_min)
        pt.show()
        
    elif method == 'tsvd' or method == 'tgsvd':
        
        R = np.zeros((m,p-1))
        
        R[:,p-2] = (beta[p-1,0] * U[:,p-1]).T

        for i in range(p-2,0,-1):
            R[:,i-2] = R[:,i-1] + (beta[i-1,0] * U[:,i-1]).T
        

        if np.isreal(np.all(beta)):
            print('real')
            q = int(np.floor(m/2))
        else:
            print('non real')
            q = int(m-1)
        
        Dfft = np.abs(np.fft.fft(R,axis=0))**2

        D = Dfft[1:q+1,:]
        

        v = np.arange(1,q+1)/q

        cp = np.zeros((q,p-1))
        dist = np.zeros((p-1,1))
              
        for k in range(0,p-1):
            cp[:,k] = np.cumsum(D[:,k])/(np.sum(D[:,k])+1e-5)
            dist[k] = np.linalg.norm(cp[:,k]-v)
        
        
        dist_min = np.min(dist)        
        reg_min = np.argmin(dist)

        pt.figure(2)
        pt.plot(cp)
        pt.plot(np.arange(0,q),cp[:,reg_min],'-r',linewidth=3)
        pt.title('Most white for k = %1.5f' %reg_min)
        pt.show()
        
        reg_param = np.arange(0,p-2)
    
    elif method == 'dsvd':
        
        reg_param = np.zeros((npoints,1))
        
        reg_param[npoints-1] = np.max([s[p-1],s[0]*smin_ratio])

        ratio = (s[0]/reg_param[npoints-1])**(1/(npoints-1))
        
        for i in range(npoints-2,-1,-1): 
            reg_param[i] = ratio * reg_param[i+1]
        
        dists = np.zeros((npoints,1))
        
        if np.isreal(np.all(beta)):
            print('real')
            q = int(np.floor(m/2))
        else:
            print('non real')
            q = int(m-1)

        cp = np.zeros((q,npoints))
    

        for i in range(0,npoints):        
            dists[i], cp[:,i] = ncpfun2(reg_param[i],s,beta,U[:,:p],1)

            # Plot selected NCPs.
        stp = int(np.around(npoints/nNCPs))

        pt.figure(2)
        pt.plot(cp[:,0:npoints:stp])


        # Find minimum.
        minG = np.min(dists)
        minGi = np.argmin(dists)
        #minGi = int(np.min(np.where(dists == dists.min())[0])) # Initial guess.
        
    
        x1 = reg_param[np.min([minGi+1,npoints-1])]
        x2 = reg_param[np.max([minGi-1,1])]


        # Minimizer.
        if x2 < x1:
            reg_min = spo.fminbound(ncpfun,x2,x1,args=(s,beta[:p],U[:,:p],1),disp=0) 
        elif x1 < x2:    
            reg_min = spo.fminbound(ncpfun,x1,x2,args=(s,beta,U[:,:p],1),disp=0)
        elif x1 == x2:
            reg_min = x1
        

        dist,cp = ncpfun3(reg_min,s,beta[:p],U[:,:p],1)
#    
        pt.plot(cp,'-r', linewidth=2)
        pt.title('Selected NCPs. Most white for $\lambda$ = %1.5f' %reg_min)
        pt.show()
        
    
    elif method == 'mtsv' or method == 'ttls':
        print('The MTSVD and TTLS methods are not supported')
        sys.exit()
    
    else:
        print('Illegal Method')
        sys.exit()
        
    return reg_min, dist, reg_param

def ncpfun3(xlambda,s,beta,U,dsvd):
    '''
    Auxiliary routine for ncp.  PCH, IMM, Dec. 30, 2007.
    '''
    
    import sys
    import numpy as np
    
    if dsvd:
        f = xlambda/(s + xlambda)
    if not dsvd:
        f = (xlambda**2)/(s**2 + xlambda**2)
    
    r1 = np.multiply(f,beta)
    r = U @ r1
     
    m = len(r)

    if np.isreal(np.all(beta)):
        q = int(np.floor(m/2))
    else:
        q = int(m-1)

    D = np.abs(np.fft.fft(r,axis=0))**2

    D = D[1:q+1]

    vvv = np.arange(1,q+1)/q
     
    cppp = np.cumsum(D)/np.sum(D)
    rest = cppp-vvv
    
    dist = np.sqrt(np.sum(np.abs(rest)**2))    

    return dist,cppp.T
        
        
def ncpfun2(xlambda,s,beta,U,dsvd):
    
    '''
    Auxiliary routine for ncp.  PCH, IMM, Dec. 30, 2007.
    '''
    
    
    import numpy as np


    if dsvd:
        f = xlambda/(s + xlambda)
    if not dsvd:
        f = (xlambda**2)/(s**2 + xlambda**2)  

    
    fb = np.multiply(f,beta)
    r = U @ fb
    #r = np.reshape(r,(-1,1))
    
    m = len(r)
    
    if np.isreal(np.all(beta)):
        q = int(np.floor(m/2))
    else:
        q = int(m-1)
    

    D = np.power(np.abs(np.fft.fft(r,axis=0)),2)
    D = D[1:q+1]

    vvv = list(np.arange(1,q+1).T/q)
    vvv = np.reshape(vvv,(-1,1))

    
    cppp = D.cumsum(axis=0)/np.sum(D)

    rest = cppp-vvv
    
    dist = np.linalg.norm(rest)

    return dist, cppp.T
    
def ncpfun(xlambda,s,beta,U,dsvd):
    
    '''
    Auxiliary routine for ncp.  PCH, IMM, Dec. 30, 2007.
    '''
        
    import sys
    import numpy as np

    if dsvd:
        f = xlambda/(s + xlambda)
    if not dsvd:
        f = (xlambda**2)/(s**2 + xlambda**2)
    
    r1 = np.multiply(f,beta)
    r = U @ r1
     
    m = len(r)

    if np.isreal(np.all(beta)):
        q = int(np.floor(m/2))
    else:
        q = int(m-1)

    D = np.abs(np.fft.fft(r,axis=0))**2

    D = D[1:q+1]

    vvv = np.arange(1,q+1)/q
     
    cppp = np.cumsum(D)/np.sum(D)

    rest = cppp-vvv
    
    dist = np.linalg.norm(rest)  
    
    return dist
    
        