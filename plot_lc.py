#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:32:14 2020

@author: fran parra-rojas
"""

def plot_lc(rho, eta, marker, ps, reg_param):
    
    import sys
    import numpy as np
    import matplotlib.pyplot as pt
    
    npp = 10
    
    if ps < 1 or ps > 2:
        print('ERROR: Illegal value of ps')
        sys.exit()
    
    n = len(rho)
    ni = int(np.rint(n/npp))
    
    pt.figure(2)
    pt.loglog(rho[1:-2],eta[1:-2])
    pt.loglog(rho,eta)
    
    if np.max(eta)/np.min(eta) > 10 or np.max(rho)/np.min(rho) > 10:
        pt.loglog(rho, eta,marker,rho[ni:ni:n],eta[ni:ni:n],'x')
    
    else:
        pt.plot(rho, eta,marker,rho[ni:ni:n],eta[ni:ni:n],'x')
        
        for k in range(ni,n,ni):
            pt.text(rho[k],eta[k],str(reg_param[k]))
    
    pt.xlabel('residual norm || A x - b ||_2')
    if ps == 1:
        pt.ylabel('solution norm || x ||_2')
    else:
        pt.ylabel('solution semi-norm || L x ||_2')
    
    pt.title('L-curve')