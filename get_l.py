#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:36:32 2019

@author: fran parra-rojas
"""

import sys
import numpy as np
import scipy.sparse as sps

def get_l( n, d ):
    
    nd = n - d
    
    if d < 0:
        print('Order d must be nonnegative')
        sys.exit()
    elif d == 0:
        print('Zero-Order regularization matrix')
        print('================================')
        L = np.matrix(np.identity(n))
        Lm = L
    elif d == 1:
        print('First-Order regularization matrix')
        print('================================')
        L = sps.diags([-1,1],[0,1],shape=(nd,n))
        Lm = L.todense()
    elif d == 2:
        print('Second-Order regularization matrix')
        print('================================')
        L = sps.diags([1,-2,1],[0,1,2],shape=(nd,n))
        Lm = L.todense()
    else:
        print('Order too high (Not yet implemented)')
        sys.exit()
        
    return Lm