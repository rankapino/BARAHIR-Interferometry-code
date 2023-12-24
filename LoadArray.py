#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:59:09 2020

@author: fran parra-rojas
"""

def LoadArray( antenna_file, Nant, antPos ):

    import numpy as np
    
    fi = open(antenna_file)
    for li,l in enumerate(fi.readlines()):
        comm=l.find('#') #   0 is the text lines and with -1 the non-text lines        
        if comm==0: # select the useful data
            l=l[:comm]
        it=l.split()  
        if len(it)>0:
            if it[0]=='ANTENNA':  
                antPos.append(list(map(float,it[1:]))) # load the antenna position
                Nant += 1
                antPos[-1][0] *= 1.e-3 # km
                antPos[-1][1] *= 1.e-3 # km
    fi.close()
    
    posarray=np.asarray(antPos) # read and change the format of the coordinates of the antenna position
    agua_x=posarray[:int(Nant),0] # coordinate of the X position
    agua_y=posarray[:int(Nant),1] # coordinate of the Y position
    
    
    return Nant, 1.0e3*posarray[:,0:2]