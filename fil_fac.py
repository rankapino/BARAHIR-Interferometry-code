#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:02:40 2020

@author: fran parra-rojas
"""

def fil_fac(s, reg_param):
    
    '''
    FIL_FAC Filter factors for some regularization methods.
    
    f = fil_fac(s,reg_param,method)
    f = fil_fac(sm,reg_param,method)  ,  sm = [sigma,mu]
    f = fil_fac(s,k,'ttls',s1,V1)
    
    Computes all the filter factors corresponding to the singular values in s
    (must be computed by the function csvd) and the regularization parameter
    reg_param, for the following methods:
        method = 'dsvd' : damped SVD or GSVD
        method = 'tsvd' : truncated SVD or GSVD
        method = 'Tikh' : Tikhonov regularization
        method = 'ttls' : truncated TLS.
    If sm = [sigma,mu] is specified (must be computed by the function cgsvd),
    then the filter factors for the corresponding generalized methods are
    computed.
    
    If method = 'ttls' then the singular values s1 and the right singular
    matrix V1 of [A,b] must also be supplied, as computed by
    [U1,s1,V1] = csvd([A,b],'full').
    
    If method is not specified, 'Tikh' is default.
    Per Christian Hansen, DTU Compute, 12/29/97.
    
    Adapted by Francisco Parra-Rojas
    '''
    
    import sys
    import numpy as np
    
    # Initialization.
    if len(s.shape) == 1:    
        p, = np.shape(s)
        ps = 1
    else:
        p, ps = np.shape(s)
        
    lr = len(reg_param)
    
    f = np.zeros((p,lr))
    
    # Check input data.
    if min(reg_param) <= 0:
        print('Regularization parameter must be positive')
        sys.exit()
        
    # Compute the filter factors.
    for j in range(0,lr):
        if ps == 1:
            f[:,j] = (s**2)/(s**2 + reg_param[j]**2)
        else:
            f[:,j] = (s[:,0]**2)/(s[:,0]**2 + reg_param[j]**2 * s[:,1]**2)
        
    return f