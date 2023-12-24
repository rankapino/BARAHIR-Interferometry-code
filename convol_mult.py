import numpy as np
from scipy.linalg import toeplitz



def matrix_to_vector(input):
    """
    Converts the input matrix to a vector by stacking the rows in a specific way explained here
    
    Arg:
    input -- a numpy matrix
    
    Returns:
    ouput_vector -- a column vector with size input.shape[0]*input.shape[1]
    """
    input_h, input_w = input.shape
    output_vector = np.zeros(input_h*input_w, dtype=input.dtype)
    # flip the input matrix up-down because last row should go first
    input = np.flipud(input) 
    for i,row in enumerate(input):
        st = i*input_w
        nd = st + input_w
        output_vector[st:nd] = row   
    return output_vector


def vector_to_matrix(input, output_shape):
    """
    Reshapes the output of the maxtrix multiplication to the shape "output_shape"
    
    Arg:
    input -- a numpy vector
    
    Returns:
    output -- numpy matrix with shape "output_shape"
    """
    output_h, output_w = output_shape
    output = np.zeros(output_shape, dtype=input.dtype)
    for i in range(output_h):
        st = i*output_w
        nd = st + output_w
        output[i, :] = input[st:nd]
    # flip the output matrix up-down to get correct result
    output=np.flipud(output)
    return output


def convolution_as_maultiplication(I, F, print_ir=False):
    """
    Performs 2D convolution between input I and filter F by converting the F to a toeplitz matrix and multiply it
      with vectorizes version of I
      By : AliSaaalehi@gmail.com
      
    Arg:
    I -- 2D numpy matrix
    F -- numpy 2D matrix
    print_ir -- if True, all intermediate resutls will be printed after each step of the algorithms
    
    Returns: 
    output -- 2D numpy matrix, result of convolving I with F
    """
    # number of columns and rows of the input 
    I_row_num, I_col_num = I.shape 

    # number of columns and rows of the filter
    F_row_num, F_col_num = F.shape

    #  calculate the output dimensions
    output_row_num = I_row_num + F_row_num - 1
    output_col_num = I_col_num + F_col_num - 1
    if print_ir: print('output dimension:', output_row_num, output_col_num)

    # zero pad the filter
    F_zero_padded = np.pad(F, ((output_row_num - F_row_num, 0),
                               (0, output_col_num - F_col_num)),
                            'constant', constant_values=0)
    if print_ir: print('F_zero_padded: ', F_zero_padded)

    # use each row of the zero-padded F to creat a toeplitz matrix. 
    #  Number of columns in this matrices are same as numbe of columns of input signal
    toeplitz_list = []
    for i in range(F_zero_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
        c = F_zero_padded[i, :] # i th row of the F 
        r = np.r_[c[0], np.zeros(I_col_num-1)] # first row for the toeplitz fuction should be defined otherwise
                                                            # the result is wrong
        toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
        toeplitz_list.append(toeplitz_m)
        if print_ir: print('F '+ str(i)+'\n', toeplitz_m)

        # doubly blocked toeplitz indices: 
    #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    c = range(1, F_zero_padded.shape[0]+1)
    r = np.r_[c[0], np.zeros(I_row_num-1, dtype=int)]
    doubly_indices = toeplitz(c, r)
    if print_ir: print('doubly indices \n', doubly_indices)

    ## creat doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

    if print_ir: print('doubly_blocked: ', doubly_blocked)

    # convert I to a vector
    vectorized_I = matrix_to_vector(I)
    if print_ir: print('vectorized_I: ', vectorized_I)
    
    # get result of the convolution by matrix mupltiplication
    result_vector = np.matmul(doubly_blocked, vectorized_I)
    if print_ir: print('result_vector: ', result_vector)

    # reshape the raw rsult to desired matrix form
    out_shape = [output_row_num, output_col_num]
    output = vector_to_matrix(result_vector, out_shape)
    if print_ir: print('Result of implemented method: \n', output)
    
    return output

# test on different examples

# fill I an F with random numbers
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 00:09:34 2020

@author: fran
"""

import os
import time
import getpass
import numpy as np
import scipy.signal as sps
import scipy.linalg as spl 
import matplotlib.pyplot as pt
import matplotlib.image as plimg
import scipy.ndimage.interpolation as spndint

time_start = time.time()

version = 9.0

print('\n')
print('==================================================================')
print('==================================================================')
print('ARA Image Software (F.C. Parra-Rojas and B. Isham) - version %s' %version)
print('\n')
print('   Inter American University of Puerto Rico - Bayamón Campus')
print('==================================================================')
print('User:', getpass.getuser())
print('Date:', time.ctime())
print('==================================================================')
print('\n')

''' inputs '''
# Please select the number of pixels
#Npix = 8
#Npix = 16
Npix = 32 # Number of pixels (Multiple of 2)
#Npix = 64
print('...............Number of Pixels > %i' %Npix)


# Please select the frequency
frequency = 5.1e6 # Hz
# frequency = 8.175e6 # Hz
# frequency = 30e6 # Hz
# frequency = 60e6 # Hz
print('...............Frequency > %.2f MHz' %(1.0e-6*frequency))



# please select the noise (0-1)
noise = 0.02 # noise (percentage)
noise = 0.0
print('...............Noise > %.2f %%' %(100 * noise))



# Please select the Test Image
flagtest = 1 # Point at zenith
#flagtest = 2 # Point at 60 degrees south
#flagtest = 3 # Point at 60 degrees east
#flagtest = 4 # Snake Eyes
#flagtest = 5 # Straight line
#flagtest = 6 # Parallel lines
#flagtest = 7 # Perpendicular lines
#flagtest = 8 # Centered square
#flagtest = 9 # Centered rectangle
#flagtest = 10 # Centered circle
#flagtest = 11 # Centered ellipse
#flagtest = 12 # Centered equilateral triangle
#flagtest = 13 # Centered rectangle triangle
#flagtest = 14 # Off center rectangle triangle
#flagtest = 15 # Off center trapezoid rectangle
flagtest = 16 # Crab Nebula
#flagtest = 17 # Lena


# Please select the Beam calculation method
flagbeam = 1 # AF
# flagbeam = 2 # BF
# flagbeam = 3 # BT

# Please select the Regularization Method
#flagreg = 1 # Modified Truncated SVD (MTSVD)
#flagreg = 2 # Truncated Gereralizated SVD (TGSVD)
#flagreg = 3 # Linear Least-Sauare (LSQLIN)
#flagreg = 4 # parallel LOG-entropy Multiplicative Algebraic Reconstruction Technique (LOGMART)
#flagreg = 5 # Kaczmarz's method
#flagreg = 6 # Truncated SVD (TSVD). Only with GCV, NCP and QOC
#flagreg = 7 # Truncated Total Least Squares (TTLS)
#flagreg = 8 # MaxEnt
#flagreg = 9 # Tikhonov
#flagreg = 10 # Lanczos Bidiagonalization method
#flagreg = 11 # Morozov Discrepancy
#flagreg = 12 # Dumped SVD
#flagreg = 13 # CLEAN

# Please select the Regularization Parameter Method
#flagparam = 1 # Generalized Cross-Validation (GCV)
#flagparam = 2 # L-Curve
#flagparam = 3 # Normalized Cumulative Periodograms (NCP)
#flagparam = 4 # Quasi-Optimality Criterion (QOC)

method = 'tikh'
param = 'gcv'
order = -1

''' Some important previous calculations '''

Nphf = int(Npix/2) # half pixel number (plot)
Nant = 0
antPos=[] # empty array for the antenna position

I_dirty = None
I_residual = None
I_model = None
clean_beam = None

wavelength = 3e8/frequency # wavelength in m

gain = 0.5 # gain (for CLEAN)
niter = 100 # iterations number (for CLEAN)
threshold = 2.*noise # minimum flux threshold to deconvolve (for CLEAN)



print('\n')
''' Test Image '''
time_start_model = time.time()

# Ring optimization depending of the Number of Pixels
if int(Npix) <= 16:
    Ro = 5
else:
    Ro = 10

import Models as MD
MAP = MD.TestImage( Npix, Nphf, Ro, flagtest  )

time_model = time.time() - time_start_model
print('Time Load Model', time_model, 'sec')
print('\n')

''' Array Load '''
time_start_array = time.time()

antenna_file = './layouts/Aguadilla.array' # Old antenna position file
antenna_file = './layouts/Aguadilla24a.array' # 24 antennas position file  
antenna_file = './layouts/Aguadilla48a.array' # 48 antenna position file
#antenna_file = './layouts/Test3antennas.array' # 3 antenna position file
    
import LoadArray as LA   
Nant, antPosmeters = LA.LoadArray( antenna_file, Nant, antPos )

print('...............Number of Antennas > %i' %Nant)
time_array = time.time() - time_start_array
print('Time Load Array', time_array, 'sec')
print('\n')


''' Beam Calculations '''
time_start_beam = time.time()
import Beam as BM
bm, uu, vv  = BM.SelectBeam( antPosmeters, wavelength, Nant, Npix, flagbeam )

time_beam = time.time() - time_start_beam
print('Time Beam Calculation', time_beam, 'sec')
print('\n')

my_result = convolution_as_maultiplication(bm,MAP + noise * np.random.randn(Npix,Npix))
print('my result: \n', my_result)
    
from scipy import signal
lib_result = signal.convolve2d(bm,MAP + noise * np.random.randn(Npix,Npix), "full")
print('lib result: \n', lib_result)

assert(my_result.all() == lib_result.all())

err = np.abs(my_result.flatten() - lib_result.flatten())

import matplotlib.pyplot as pt
pt.figure()
pt.plot(err)


