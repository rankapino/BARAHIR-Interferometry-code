#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 21:00:47 2020

@author: fran parra-rojas
"""

def UVPlane( Nant, antPosmeters, wavelength ):
    
    import numpy as np
    nbaselines = Nant*(Nant-1)/2    

    uu = np.zeros(int(2*nbaselines))
    vv = np.zeros(int(2*nbaselines))

    baselines1s = np.zeros((int(nbaselines),2))

    k = 0
    for i in list(range(Nant)):
        for j in list(range(i+1,Nant)):
            
            baselines1s[k,0] = (antPosmeters[i][0]-antPosmeters[j][0])/wavelength # wavelengths # x 
            baselines1s[k,1] = (antPosmeters[i][1]-antPosmeters[j][1])/wavelength # wavelengths # y 
            k = k+1

    # reflected baselines - units of wavelength 
    uu = np.append(baselines1s[:,0],-baselines1s[:,0]) # wavelengths 
    vv = np.append(baselines1s[:,1],-baselines1s[:,1]) # wavelengths
    
    return uu,vv