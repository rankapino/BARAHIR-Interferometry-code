#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:10:36 2020

@author: fran parra-rojas
"""

def gcv_fun(xlambda,s2,beta,delta0,mn):
    
    import numpy as np
    f = (xlambda**2)/(s2 + xlambda**2)
    G = (np.linalg.norm(f * beta.T)**2 + delta0)/(mn + np.sum(f))**2