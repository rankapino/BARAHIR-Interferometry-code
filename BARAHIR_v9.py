#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 00:09:34 2020

@author: fran parra-rojas
"""

import os
import time
import getpass
import numpy as np
import scipy.signal as sps
import scipy.linalg as spl 
import numpy.linalg as npl 
import matplotlib.pyplot as pt
import matplotlib.image as plimg
import scipy.ndimage.interpolation as spndint

time_start = time.time()

version = 9.0

print('\n')
print('==================================================================')
print('==================================================================')
print('                         BARAHIR Image Code                       ')
print('      Bayamón-Aguadilla Radio Array HIgh-Resolution Image Code    ')
print('                   F.C. Parra-Rojas and B. Isham                  ') 
print('                           version %s' %version                    )
print('\n')
print('   Inter American University of Puerto Rico - Bayamón Campus')
print('==================================================================')
print('User:', getpass.getuser())
print('Date:', time.ctime())
print('==================================================================')
print('\n')

''' inputs '''
# Please select the number of pixels
# Npix = 8
# Npix = 16
Npix = 32 # Number of pixels (Multiple of 2)
#Npix = 64
print('...............Number of Pixels > %i' %Npix)


# Please select the frequency
frequency = 5.1e6 # Hz
# frequency = 8.175e6 # Hz
# frequency = 30e6 # Hz
#frequency = 60e6 # Hz
print('...............Frequency > %.2f MHz' %(1.0e-6*frequency))


# please select the noise (0-1)
noise = 0.02 # noise (percentage)
# noise = 0.0
print('...............Noise > %.2f %%' %(100 * noise))


# Please select the Test Image
flagtest = 1 # Point at zenith
flagtest = 2 # Point at 60 degrees south
flagtest = 3 # Point at 60 degrees east
flagtest = 4 # Snake Eyes
flagtest = 5 # Straight line
flagtest = 6 # Parallel lines
#flagtest = 7 # Perpendicular lines
#flagtest = 8 # Centered square
#flagtest = 9 # Centered rectangle
#flagtest = 10 # Centered circle
#flagtest = 11 # Centered ellipse
#flagtest = 12 # Centered equilateral triangle
#flagtest = 13 # Centered rectangle triangle
#flagtest = 14 # Off center rectangle triangle
#flagtest = 15 # Off center trapezoid rectangle
#flagtest = 16 # Crab Nebula
#flagtest = 17 # Lena


# Please select the Beam calculation method
flagbeam = 1 # AF
#flagbeam = 2 # BF
#flagbeam = 3 # BT


# Please select the Regularization Method
# Non-Iterative Methods
#method = 'mtsvd' # Modified Truncated SVD Method (MTSVD) 
#method = 'tgsvd' # Truncated Gereralizated SVD Method (TGSVD)
#method = 'tsvd' # Truncated SVD Method (TSVD). Only with GCV, NCP and QOC
#method = 'ttls' # Truncated Total Least Squares Method (TTLS)
method = 'tikh' # Tikhonov Method
#method = 'disc' # Morozov Discrepancy Method
#method = 'dsvd' # Dumped SVD Method

# Other Methods
#method = 'lsqlin' # Linear Least-Sauare Method (LSQLIN)
#method = 'logmart' # parallel LOG-entropy Multiplicative Algebraic Reconstruction Technique (LOGMART)
#method = 'kaczmarz' # Kaczmarz's Method
#method = 'maxent' # Maximum Entropy Method
#method = 'lanczos' # Lanczos Bidiagonalization Method
#method = 'clean' # CLEAN Method (Hogbom Algorithm)


# please selec the method to get the regularization parameter
#param = 'gcv' # Generalized Cross-Validation (GCV)
param = 'ncp' # Normalized Cumulative Periodograms (NCP)
#param = 'lcurve' # L-Curve
#param = 'qoc' # Quasi-Optimality Criterion (QOC)

order = -1 # Zero order
#order = 2


#karray = 10**(np.linspace(2,3,num=10)) # only for MTSVD  or TTLS
#karray = 10 # for Lanczos

karray = 10**(np.linspace(1,2,num=10)) # only for Discrepancy
#karray = 10**(np.linspace(-7,1,num=10)) # only for maxent


''' Some important previous calculations '''

Nphf = int(Npix/2) # half pixel number (plot)
Nant = 0
antPos=[] # empty array for the antenna position

I_dirty = None
I_residual = None
I_model = None
clean_beam = None

wavelength = 3e8/frequency # wavelength in m

gain = 0.0001 # gain (for CLEAN)
niter = 10000 # iterations number (for CLEAN)
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

''' Dirty Map '''
time_start_convolution = time.time()
DirtyMap = sps.convolve2d(bm,MAP + noise * np.random.randn(Npix,Npix))

time_convolution = time.time() - time_start_convolution
print('Time Convolution Calculation', time_beam, 'sec')
print('\n')

''' Toeplitz Matrix '''
time_start_Top = time.time()
import Toep as TP
Top = TP.Toep(bm)
time_Top = time.time() - time_start_Top
print('Time Toeplitz Calculation', time_Top, 'sec')
print('\n')

kappa = npl.cond(Top)
print('...............Condition Number > %.2f' %(kappa))
if kappa > 1.5:
    print('Be carefull, the problem is Ill-conditioned.')
    print('A small error in Dirty Map can produce a big error in the Solution.')
else:
    print('The problem is Well-conditioned.')
print('\n')

import RecImage2 as RI
im_rec = RI.RecImage( Top, DirtyMap.flatten(), method, order, param, karray, gain, niter, threshold, Npix, bm )


''' Plot '''
import PlotSol as PS
PS.plotSol( Npix, frequency, antPosmeters, Nant, bm, DirtyMap, MAP, im_rec, method, order )