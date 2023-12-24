#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:01:55 2020

@author: fran parra-rojas
"""

def Toep(Beam):
    
    import numpy as np
    import scipy.linalg as spl
    
    # number of columns and rows of the input (same of Beam)
    I_row_num, I_col_num = Beam.shape
    
    # number of columns and rows of the Beam
    F_row_num, F_col_num = Beam.shape
    
    # calculate the output dimensions
    output_row_num = I_row_num + F_row_num - 1
    output_col_num = I_col_num + F_col_num - 1
    
    
    # zero pad the filler
    F_zero_padded = np.pad(Beam, ((output_row_num - F_row_num, 0),(0, output_col_num - F_col_num)), 'constant', constant_values = 0)
    
    
    toeplitz_list = []
    # iterate from last row to the first row
    for i in range(F_zero_padded.shape[0]-1, -1, -1):
        c = F_zero_padded[i,:] # copy i'th row of the F to c
        r = np.r_[c[0], np.zeros(I_col_num-1)]
        
        # toeplitz function
        toeplitz_m = spl.toeplitz(c,r)
        toeplitz_list.append(toeplitz_m)
        #print('F_' + str(i) + '\n', toeplitz_m)
    
    
    c = range(1, F_zero_padded.shape[0]+1)
    r = np.r_[c[0], np.zeros(I_row_num-1, dtype=int)]
    
    doubly_indices = spl.toeplitz(c,r)
    #print('doubly_indices_\n', doubly_indices)
    
    
    # shape of one of those small toeplitz matrices
    h = toeplitz_m.shape[0] * doubly_indices.shape[0]
    w = toeplitz_m.shape[1] * doubly_indices.shape[1]
    
    doubly_blocked_shape = [h,w]
    doubly_blocked = np.zeros(doubly_blocked_shape)
    
    # tile the toeplitz matrix
    b_h, b_w = toeplitz_m.shape # hight and width of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i:end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]
    
    #print(doubly_blocked)
    
    return doubly_blocked
    