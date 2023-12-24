#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:00:16 2020

@author: fran parra-rojas
"""

def ncpfun(xlambda,s,beta,U):
    
    import numpy as np

    f = (xlambda**2)/(s**2 + xlambda**2)

       
    r = U @ (f * beta)

     
    m = len(r)
    
    if np.isreal(np.all(beta)):
        q = int(np.floor(m/2))
    else:
        q = int(m-1)
    
    
    D = np.abs(np.fft.fft(r,axis=0))**2
    D = D[1:q+1]


    vvv = list(np.arange(1,q+1).T/q)
   
    
    cppp = D.cumsum(axis=0)/np.sum(D)
    

    rest = cppp-vvv
    
    dist = np.sqrt(np.sum(np.abs(rest)**2))    

#    print(dist)
#    print(np.shape(cppp))
    return dist, cppp