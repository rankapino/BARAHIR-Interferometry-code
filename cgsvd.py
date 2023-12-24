#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:15:45 2019

@author: fran parra-rojas
"""



import sys
import scipy
import numpy as np
import gsvd as gsvd
from numpy.linalg import inv

def cgsvd(A,B):

    m,n = A.shape
    p,n1 = B.shape

    if n1 != n:
        print('Error: Number of columns in A and L must be the same')
        sys.exit()

    if m+p < n:
        print('Error: Dimensions must satisfy m+p > n')
        sys.exit()


    U,V,W,C,S = gsvd.gsvd(A,B,0)
    
#    import validation as vad
#    vad.ValidationTests( A, B, U, V, W, C, S )
    
    if m >= n:
        q = min(p,n)
#    print(np.concatenate([np.diagonal(C[0:q,0:q]),np.diagonal(S[0:q,0:q])]))
        #sm = np.concatenate([np.diagonal(C[0:q,0:q]),np.diagonal(S[0:q,0:q])])
        sm = np.column_stack([np.diagonal(C[0:q,0:q]),np.diagonal(S[0:q,0:q])])
        X = inv(W.T)
    else:
        sm = np.column_stack([np.diagonal(C[0:m+p-n,n-m:p]),np.diagonal(S[n-m:p,n-m:p])])
        #sm = np.concatenate([np.diagonal(C[0:m+p-n,n-m:p]),np.diagonal(S[n-m:p,n-m:p])])
        X = inv(W.T)
        X = X[:,n-m:n]

    W = W.T

    return U, sm, X, V, W 
