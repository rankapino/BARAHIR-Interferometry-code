#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:27:19 2019

@author: fran parra-rojas
"""

import numpy as np
import sys
import scipy
import scipy.linalg as scp
import scipy.sparse

def csd(Q1,Q2):
    '''CSD Cosine-Sine Decomposition
    [U,V,Z,C,S] = csd(Q1,Q2)
    
    Given Q1 and Q2 such that Q1'*Q1 + Q2'*Q2 = I, the
    C-S Decomposition is a joint factorization of the form
        Q1 = U*C*Z' and Q2=V*S*Z'
    where U, V, and Z are orthogonal matrices and C and S
    are diagonal matrices (not necessarily square) satisfying
        C'*C + S'*S = I
    '''
    m,p = Q1.shape
    n,pb = Q2.shape
    if pb != p:
        print ("gsvd Matrix Column Mismatch : Matrices must have the same number of columns")
        sys.exit()
    if m < n:
        V,U,Z,S,C = csd(Q2,Q1)
        j = np.arange(p-1,-1,-1)
        C = C[:,j]
        S = S[:,j]
        Z = Z[:,j]
        m = min([m,p])
        i = np.arange(m-1,0,-1)
        C[0:m-1,:] = C[i,:]; 
        U[:,0:m-1] = U[:,i];
        n = min([n,p])
        i = np.arange(n-1,-1,-1)
        S[0:n,:] = S[i,:]
        V[:,0:n] = V[:,i];
        return V,U,Z,S,C
        # Henceforth, n <= m.

    U,C0,Z0 = np.linalg.svd(Q1);
    Z = Z0.T
    
    C = scp.diagsvd(C0,(Q1.shape)[0],(Q1.shape)[1])
    
    q = min([m,p])
    i = np.arange(0,q)
    j = np.arange(q-1,-1,-1)   
    C[i,i]=C[j,j]
    U[:,i] = U[:,j]
    Z[:,i] = Z[:,j]

#    Z = np.fliplr(np.flipud(Z)).T
    S = np.dot(Q2,Z)
    if q == 0:
        k = 0
    elif m < p:
        k = n;
    else:
        CC = np.array(np.diag(C))
        if CC[CC<= 1/np.sqrt(2)] is not None:
            V = np.identity(S.shape[0])
            S = np.dot(V.T,S)
        else:
            k = max([0,CC[CC <= 1/np.sqrt(2)].argmax()])
            V,_ = scipy.linalg.qr(S[:,0:k]) ## THis works when k=0. Test when k > 0.
            S = np.dot(V.T,S)
            r = min([k,m])
            S[:,0:r]=diagf(S[:,0:r])
    if m == 1 and p > 1:
        S[0,0] = 0
#
    
    if CC[CC<= 1/np.sqrt(2)] is not None:
        r = min([n,p])
        i=np.arange(0,n)
        j=np.arange(0,r)
        UT, ST0, VT0 = np.linalg.svd(slice_matrix(S,i,j))
        VT = VT0.T
        ST = scp.diagsvd(ST0,len(ST0),len(ST0))  
        S[0:n,0:r] = ST
        C[:,j] = np.dot(C[:,j],VT)
        V[:,i] = np.dot(V[:,i],UT)
        Z[:,j] = np.dot(Z[:,j],VT)
        i = np.arange(0,q)
        Q,R = scipy.linalg.qr(C[0:q,0:r])
        C[0:q,0:r] = diagf(R)
        U[:,i] = np.dot(U[:,i],Q)
    elif k < min([n,p]):
        r = min([n,p])
        i=np.arange(k+1,n)
        j=np.arange(k+1,r)
        
        UT, ST0, VT = np.linalg.svd(slice_matrix(S,i,j))
        ST = scp.diagsvd(ST0,len(ST0),len(ST0))  
        if k > 0: 
            S[0:k,j] = 0
    
        S[k+1:n,k+1:r] = ST
        C[:,j] = np.dot(C[:,j],VT)
        V[:,i] = np.dot(V[:,i],UT)
        Z[:,j] = np.dot(Z[:,j],VT)
        i = np.arange(k,q)
        t = np.arange(k+1,q)
        Q,R = scipy.linalg.qr(C[k+1:n+2,k+1:r])
        C[k+1:n+2,k+1:r] = diagf(R)
        U[:,t] = np.dot(U[:,t],Q)
    
#    if m < p:
#        # Diagonalize final block of S and permute blocks.
#        q = min(scipy.sparse.lil_matrix(abs(diagk(C,0))>10*m*np.spacing(1)).getnnz(), 
#            scipy.sparse.lil_matrix(abs(diagk(S,0))>10*n*np.spacing(1)).getnnz())

#        i = np.arange(q,n)
#        j = np.arange(m,p)
#        # At this point, S(i,j) should have orthogonal columns and the
#        # elements of S(:,q+1:p) outside of S(i,j) should be negligible.
#        Q,R = scipy.linalg.qr(S[q:n,m:p])
#        S[:,q:p-1] = 0
#        S[q:n,m:p] = diagf(R)
#        V[:,i] = np.dot(V[:,i],Q)
#        if n > 1:
#            i=np.concatenate([np.arange(q,q+p-m),np.arange(0,q),np.arange(q+p-m,n)])
#        else:
#            i = 1
#        j = np.concatenate([np.arange(m,p),np.arange(0,m)])
#        t = np.arange(0,len(j))
#        C[:,[t]] = C[:,[j]]
#        del t
#        S = S[q:n,m:p]
#        Z = Z[:,j]
#        V = V[:,i]
#
#    if n < p:
#        # Final block of S is negligible.
#        S[:,n:p] = 0;

#    # Make sure C and S are real and positive.
    U,C = diagp(U,C,max([0,p-m]))
    C = C.real
    V,S = diagp(V,S,0)
    S = S.real
#
    return U, V, Z, C, S

# <codecell>

def add_zeros(C,Q):
    '''ADD_ZEROS returns the vector C padded with zeros to be the same size as matrix Q. 
    The values of C will be along the diagonal.
    
    USAGE: add_zeros(C,Q)
    '''
    assert C.shape > 1
    assert Q.shape > 1
    m,p=Q.shape
    #n,pb=C.shape
    toto = np.zeros((m,p))
    toto[0:min(m,p),0:min(m,p)]=np.diag(C)
    C=toto
    return C

# <codecell>

def diagk(X,k):
    '''DIAGK K-th matrix diagonal.
    DIAGK(X,k) is the k-th diagonal of X, even if X is a vector.'''
    if min(X.shape)> 1:
        D = np.diag(X,k)
    elif 0 <= k and 1+k <= X.shape[1]:
        D = X(1+k)
    elif k < 0 and 1-k <= X.shape[0]:
        D = X(1-k)
    else:
        D = []
    return D

# <codecell>

def diagf(X):
    ''' DIAGF Diagonal force.
    X = DIAGF(X) zeros all the elements off the main diagonal of X.
    '''
    X = np.triu(np.tril(X))
    return X

# <codecell>

def diagp(Y,X,k):
    ''' DIAGP Diagonal positive.
    Y,X = diagp(Y,X,k) scales the columns of Y and the rows of X by
    unimodular factors to make the k-th diagonal of X real and positive.
    '''
    D = diagk(X,k)
    j = [item for item, a in enumerate(D) if a.real < 0 or a.imag != 0]
    D = np.diag(np.divide(D[j].conjugate(),abs(D[j]))) ### CHECK TYPE OF DIVISION HERE!!!
    Y[:,j] = np.dot(Y[:,j],D.T)
    X[j,:] = np.dot(D,X[j,:])
    return Y, X

# <codecell>

def trim_matrix_col(Q,p):
    '''TRIM_MATRIX_COL trims the output of the 
    Q matrix output of the qr function column-wise to the number of columns of 
    the input matrices to scipy.linalg.qr to 
    match the format of the Matlab qr() function and returns Q.
    
    USAGE trim_matrix(Q,p)
    where 
    Q is the Q output matrix of scipy.linalg.gr
    p is the number of columns of the input matrices to scipy.linalg.gr
    '''
    Q=Q[:,0:p]
    return Q

# <codecell>

def trim_matrix_row(R,p):
    '''TRIM_MATRIX_ROW trims the output of the 
    R matrix output of the qr function row-wise to the number of rows of 
    the input matrices to scipy.linalg.qr to 
    match the format of the Matlab qr() function and returns R.
    
    USAGE trim_matrix(R,p)
    where 
    R is the R output matrix of scipy.linalg.gr
    p is the number of columns of the input matrices to scipy.linalg.gr
    '''
    R=R[0:p,:]
    return R

# <codecell>

def slice_matrix(X,i,j):
    '''SLICE_MATRIX returns X sliced to the 
    vector of i row indices and j column indices
    
    USAGE
    '''
    assert len(i) > 1
    assert len(j) > 1
    X=X[i,:]
    X=X[:,j]
    return X
