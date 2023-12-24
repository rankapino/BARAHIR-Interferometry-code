#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:59:15 2020

@author: fran parra-rojas
"""

def RegParam( flagparam, regtool ):
    
    if flagparam == 1:
        import gcv as GCV
        reg_min, G, reg_param = GCV.gcv( U1, sm, DirtyMap, regtool )
    
    
    return reg_min