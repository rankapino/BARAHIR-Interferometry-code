#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:56:15 2020

@author: fran parra-rojas
"""

import numpy as np
from scipy import optimize


def subtractPSF( img, psf, l, m, flux, gain ):
    """Subtract the PSF (attenuated by the gain and flux) centred at (l,m) from an image"""
    
    #get the half lengths of the PSF
    if (psf.shape[0] % 2 == 0): psfMidL = psf.shape[0]/2 #even
    else: psfMidL = (psf.shape[0]+1)/2 #odd
    if (psf.shape[1] % 2 == 0): psfMidM = psf.shape[1]/2 #even
    else: psfMidM = (psf.shape[1]+1)/2 #odd
    
    #determine limits of sub-images
    #starting m
    if m-psfMidM < 0:
        subM0 = 0
        subPSFM0 = psfMidM-m
    else:
        subM0 = m-psfMidM
        subPSFM0 = 0
    #starting l
    if l-psfMidL < 0:
        subL0 = 0
        subPSFL0 = psfMidL-l
    else:
        subL0 = l-psfMidL
        subPSFL0 = 0
    #ending m
    if img.shape[1] > m+psfMidM:
        subM1 = m+psfMidM
        subPSFM1 = psf.shape[1]
    else:
        subM1 = img.shape[1]
        subPSFM1 = psfMidM + (img.shape[1]-m)
    #ending l
    if img.shape[0] > l+psfMidL:
        subL1 = l+psfMidL
        subPSFL1 = psf.shape[0]
    else:
        subL1 = img.shape[0]
        subPSFL1 = psfMidL + (img.shape[0]-l)
    
    #select subset of image
    #subImg = img[subL0:subL1, subM0:subM1]
    #select subset of PSF
    subPSF = psf[int(subPSFL0):int(subPSFL1), int(subPSFM0):int(subPSFM1)]
    
    #subtract PSF centred on (l,m) position
    img[int(subL0):int(subL1), int(subM0):int(subM1)] -= flux * gain * psf[int(subPSFL0):int(subPSFL1), int(subPSFM0):int(subPSFM1)]
    return img

def gauss2D( x, y, amp, meanx, meany, sigmax, sigmay ):
    """2D Gaussian Function"""
    gx = -(x - meanx)**2/(2*sigmax**2)
    gy = -(y - meany)**2/(2*sigmay**2)
    
    return amp * np.exp( gx + gy)

def err( p, xx, yy, data ):
    """Error function for least-squares fitting"""
    return gauss2D(xx.flatten(), yy.flatten(), *p) - data.flatten()

def idealPSF( psfImg ):
    """Determine the ideal PSF size based on the observing PSF doing a simple 2D Gaussian least-squares fit"""
    xx, yy = np.meshgrid(np.arange(0, psfImg.shape[0]), np.arange(0, psfImg.shape[1]))
    # Initial estimate: PSF should be amplitude 1, and usually imaging over sample the PSF by 3-5 pixels
    params0 = 1., psfImg.shape[0]/2., psfImg.shape[1]/2., 3., 3.
    params, pcov, infoDict, errmsg, sucess = optimize.leastsq(err, params0, \
                            args=(xx.flatten(), yy.flatten(), psfImg.flatten()), full_output=1)
    #fwhm = [2.*np.sqrt(2.*np.log(2.)) * params[3], 2.*np.sqrt(2.*np.log(2.)) * params[4]]
    return params



def restoreImg( skyModel, residImg, params ):
    """Generate a restored image from a deconvolved sky model, residual image, ideal PSF params"""
    mdlImg = np.zeros_like(residImg)
    for l,m, flux in skyModel:
        mdlImg[l,m] += flux
    
    #generate an ideal PSF image
    psfImg = np.zeros_like(residImg)
    xx, yy = np.meshgrid(np.arange(0, psfImg.shape[0]), np.arange(0, psfImg.shape[1]))
    psfImg = gauss2D(xx, yy, params[0], params[1], params[2], params[3], params[4])
    
    #convolve ideal PSF with model image
    sampFunc = np.fft.fft2(psfImg) #sampling function
    mdlVis = np.fft.fft2(mdlImg) #sky model visibilities
    sampMdlVis = sampFunc * mdlVis #sampled sky model visibilities
    #convImg = np.abs(np.fft.fftshift(np.fft.ifft2(sampMdlVis))) + residImg #sky model convolved with PSF
    convImg = np.abs(np.fft.fftshift(np.fft.ifft2(sampMdlVis))) + residImg #sky model convolved with PSF
    
    #return mdlImg + residImg
    return convImg

def hogbom( dirtyImg, psfImg, gain, niter, fthresh ):
    #Initalization
    skyModel = [] #initialize empty model
    residImg = np.copy(dirtyImg) #copy the dirty image to the initial residual image
    i = 0 #number of iterations
    
    #CLEAN iterative loop
    while np.max(residImg) > fthresh and i < niter:
        lmax, mmax = np.unravel_index(residImg.argmax(), residImg.shape) #get pixel position of maximum value
        fmax = residImg[lmax, mmax] #flux value of maximum pixel
        print('iter %i, (l,m):(%i, %i), flux: %f'%(i, lmax, mmax, fmax))
        residImg = subtractPSF(residImg, psfImg, lmax, mmax, fmax, gain)
        skyModel.append([lmax, mmax, gain*fmax])
        i += 1
    
    return residImg, skyModel