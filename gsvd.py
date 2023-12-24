#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:45:53 2019

@author: fran parra-rojas
"""


def gsvd(A,B,*args):

    import numpy as np
    import scipy
    import csd1 as csd

    m,p = A.shape
    n,pb = B.shape
    
    if pb != p:
        print("gsvd Matrix Column Mismatch : Matrices must have the same number of columns")
        sys.exit()
    QA = []
    QB = []
    for arg in args:
        if arg == 0:
            if m > p:
                QA,A = scipy.linalg.qr(A,mode='economic')
                DA = np.diagonal(A,0)
                j = [item for item, a in enumerate(DA) if a.real < 0 or a.imag != 0]
                DA = np.diag(np.divide(DA[j].conjugate(),abs(DA[j]))) ### CHECK TYPE OF DIVISION HERE!!!
                QA[:,j] = np.dot(QA[:,j],DA.T)
                A[j,:] = np.dot(DA,A[j,:])
                m = p
            if n > p:
                QB,B = scipy.linalg.qr(B,mode='economic')
                DB = np.diagonal(A,0)
                j = [item for item, a in enumerate(DB) if a.real < 0 or a.imag != 0]
                DB = np.diag(np.divide(DB[j].conjugate(),abs(DB[j]))) ### CHECK TYPE OF DIVISION HERE!!!
                QB[:,j] = np.dot(QB[:,j],DB.T)
                B[j,:] = np.dot(DB,B[j,:])
                n = p
#


    Q, R = scipy.linalg.qr(np.concatenate([A,B]),mode='economic')
    
        
    Q1 = Q[0:m,:]
    Q2 = Q[m:m+n,:]
    

#m,p = Q1.shape
#n,pb = Q2.shape
#
#if pb != p:
#    print ("gsvd Matrix Column Mismatch : Matrices must have the same number of columns")
#    sys.exit()
#if m < n:
#    V,U,Z,S,C = csd.csd(Q2,Q1)
#    j = np.arange(p-1,-1,-1)
#    C = C[:,j]
#    S = S[:,j]
#    Z = Z[:,j]
#    m = min([m,p])
#    i = np.arange(m-1,0,-1)
#    C[0:m-1,:] = C[i,:]; 
#    U[:,0:m-1] = U[:,i];
#    n = min([n,p])
#    i = np.arange(n-1,-1,-1)
#    S[0:n,:] = S[i,:]
#    V[:,0:n] = V[:,i];


    

    U,V,Z,C,S=csd.csd(Q1,Q2)
    
    
    
    X = np.dot(R.T,Z)
    
    if np.shape(QA)[0] != 0:           
        U = np.dot(QA,U)
    if np.shape(QB)[0] != 0:
        V = np.dot(QB,V)

#    if QA.size != 0:
#        U = np.dot(QA,U)
#    if QB.size != 0:
#        V = np.dot(QB,V)
    
    return U,V,X,C,S

