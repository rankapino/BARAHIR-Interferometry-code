#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:46:28 2020

@author: fran parra-rojas
"""

import os
import sys
import numpy as np
import matplotlib.image as plimg
import scipy.ndimage.interpolation as spndint

def TestImage( Npix, Nphf, outer, flagtest ):

    if flagtest == 16:
        MAP = LoadModel( './models/Nebula.model', Npix, Nphf )
        print('...............Test Model > Crab Nebula')
    elif flagtest == 17:
        MAP = LoadModel( './models/lena.model', Npix, Nphf )
        print('...............Test Model > Lena')
    else:
        MAP = GeometricFigures( Npix, outer, flagtest )
    
    return MAP

def LoadModel( model_file, Npix, Nphf ):
    if len(model_file)==0:
        print("\n\nModel file %s does not exist!\n\n"%model_file)
        sys.exit()

    if len(model_file)>0: 
        if not os.path.exists(model_file):
            print("\n\nModel file %s does not exist!\n\n"%model_file)
            sys.exit()
        else:
            models = []
            imfiles = []
            fi = open(model_file)
            for li,l in enumerate(fi.readlines()):
                comm = l.find('#')
                if comm==0:
                    l = l[:comm]     
                it = l.split()
                if len(it)>0:
                    if it[0]=='IMAGE':
                        imfiles.append([str(it[1]),float(it[2])])
          
            if len(models)+len(imfiles)==0:
                print("\n\nThere should be at least 1 model component!\n\n")
      

    fi.close()

    modelim = [np.zeros((Npix,Npix),dtype=np.float32) for i in [0,1]]
    modelimTrue = np.zeros((Npix,Npix),dtype=np.float32)

    for imfile in imfiles:
        if not os.path.exists(imfile[0]):
            imfile[0] = os.path.join(imfile[0])
            if not os.path.exists(imfile[0]):
                print('File %s does NOT exist. Cannot read the model!'%imfile[0]) 

        Np4 = int(Npix/4)
    # Lee la imagen y la convierte a flotante 32
        img = plimg.imread(imfile[0]).astype(np.float32)
        dims = np.shape(img)
        d3 = min(2,dims[2])
        d1 = float(max(dims))
    # calcula la media de las tres matrices (axis=2), la 0, 1 y 2 ya que la 3 es vacia
        avimg = np.average(img[:,:,:d3],axis=2)
    # normaliza la imagen de modo que el mínimo sea 0 y el maximo 1
        avimg -= np.min(avimg)
        avimg *= imfile[1]/np.max(avimg)
        if d1 == Nphf:
            sh0 = (Nphf-dims[0])/2
            sh1 = (Nphf-dims[1])/2
            modelimTrue[sh0+Np4:sh0+Np4+dims[0], sh1+Np4:sh1+Np4+dims[1]] += zoomimg
        else:
# hace zoom (escalado, no amplia ni disminuye) de la imagen a traves de 
#interpolacion de splines de orden 3 con un factor de zoom a cada eje de Nphf/d1
            zoomimg = spndint.zoom(avimg,float(Nphf)/d1)
            zdims = np.shape(zoomimg)
            zd0 = min(zdims[0],Nphf)
            zd1 = min(zdims[1],Nphf)
            sh0 = round((Nphf-zdims[0]))
            sh1 = round((Nphf-zdims[1]))
    # en modelimTrue que es de 32x32 mete la imagen zoomimg que es 15x16
            modelimTrue[sh0+Np4:sh0+Np4+zd0, sh1+Np4:sh1+Np4+zd1] += zoomimg[:zd0,:zd1]

# obliga a que los valores negativos sean cero
    modelimTrue[modelimTrue<0.0] = 0.0

# es un array de 2x32x32 donde en la primera hora mete a modelimTrue y en la segunda la deja
# a cero como base de la imagen
    modelim[0][:] = modelimTrue
    Testmodel = modelim[0]
    
    return Testmodel


def GeometricFigures( imSize, outer, flaggeo ):
    """Return a circular sampling map of size [imgSize, imgSize]
    imgSize: image size in pixels
    outer: outer radius (in pixels) to exclude sampling above
    """
    if flaggeo == 1:
        sampling = np.zeros((imSize,imSize))
        sampling[int(imSize/2),int(imSize/2)] = 1.0
        print('...............Geometric Model > Point at zenith')
    
    elif flaggeo == 2:
        sampling = np.zeros((imSize,imSize))    
        sampling[int(imSize/2 + round(imSize/3)),int(imSize/2)] = 1.0
        print('...............Geometric Model > Point at 60 degrees south')
    
    elif flaggeo == 3:
        sampling = np.zeros((imSize,imSize))    
        sampling[int(imSize/2),int(imSize/2 + round(imSize/3))] = 1.0
        print('...............Geometric Model > Point at 60 degrees east')
    
    elif flaggeo == 4:
        sampling = np.zeros((imSize,imSize))    
        sampling[int(imSize/2),int(imSize/2 + round(imSize/5))] = 1.0
        sampling[int(imSize/2),int(imSize/2 - round(imSize/5))] = 1.0
        print('...............Geometric Model > Snake Eyes')
    
    elif flaggeo == 5:
        sampling = np.zeros((imSize,imSize))    
        sampling[int(imSize/2),int(imSize/2 - round(imSize/3)):int(imSize/2 + round(imSize/3))] = 1.0
        print('...............Geometric Model > Straight line')
    
    elif flaggeo == 6:
        sampling = np.zeros((imSize,imSize))
        sampling[int(imSize/2 - round(imSize/5)),int(imSize/2 - round(imSize/3)):int(imSize/2 + round(imSize/3))] = 1.0
        sampling[int(imSize/2 + round(imSize/5)),int(imSize/2 - round(imSize/3)):int(imSize/2 + round(imSize/3))] = 1.0
        print('...............Geometric Model > Parallel lines')
        
    elif flaggeo == 7:
        sampling = np.zeros((imSize,imSize))    
        sampling[int(imSize/2),int(imSize/2 - round(imSize/3)):int(imSize/2 + round(imSize/3))] = 1.0
        sampling[int(imSize/2 - round(imSize/3)):int(imSize/2 + round(imSize/3)),int(imSize/2)] = 1.0
        print('...............Geometric Model > Perpendicular lines')
    
    elif flaggeo == 8:
        sampling = np.zeros((imSize,imSize), dtype='float')
        sampling[int(imSize/2 - round(imSize/5)):int(imSize/2 + round(imSize/5)),int(imSize/2 - round(imSize/5)):int(imSize/2 + round(imSize/5))] = 1.0
        print('...............Geometric Model > Centered square')
    
    elif flaggeo == 9:
        sampling = np.zeros((imSize,imSize), dtype='float')
        sampling[int(imSize/2 - round(imSize/5)):int(imSize/2 + round(imSize/5)),int(imSize/2 - round(imSize/4)):int(imSize/2 + round(imSize/4))] = 1.0
        print('...............Geometric Model > Centered rectangle')
    
    elif flaggeo == 10:
        sampling = np.zeros((imSize,imSize), dtype='float')
        ones = np.ones((imSize,imSize), dtype='float')
        xpos, ypos = np.mgrid[0:imSize,0:imSize]
        radius = np.sqrt((xpos - imSize/2)**2. + (ypos - imSize/2)**2.)
        sampling = np.where((outer >= radius), ones, sampling)
        print('...............Geometric Model > Centered circle')
    
    elif flaggeo == 11: 
        a = imSize/2
        b = imSize/4
        x = np.linspace(-imSize, imSize, imSize)
        y = np.linspace(-imSize, imSize, imSize)
        xgrid, ygrid = np.meshgrid(x, y)
        ellipse = xgrid**2 / a**2 + ygrid**2 / b**2
        sampling = np.zeros((imSize,imSize), dtype=np.int32)
        sampling[ellipse < 1.0] = 1.0
        print('...............Geometric Model > Centered ellipse')
        
    elif flaggeo == 12:
        sampling = np.zeros((imSize,imSize)) 
        for i in range(0,int(imSize/2)):
            sampling[int(imSize/2 + round(imSize/8))-i,int(imSize/2 - round(imSize/3)+i):int(imSize/2 + round(imSize/3)-i)] = 1.0
        print('...............Geometric Model > Centered equilateral triangle')
        
    elif flaggeo == 13:
        sampling = np.zeros((imSize,imSize)) 
        for i in range(0,int(imSize/2)):
            sampling[int(imSize/2 + round(imSize/8))-i,int(imSize/2):int(imSize/2 + round(imSize/3)-i)] = 1.0
        print('...............Geometric Model > Centered rectangle triangle')
        
    elif flaggeo == 14:
        sampling = np.zeros((imSize,imSize)) 
        for i in range(0,int(imSize/2)):
            sampling[int(imSize/2 + round(imSize/8))-i,int(imSize/8):int(imSize/8 + round(imSize/3)-i)] = 1.0
        print('...............Geometric Model > Off center rectangle triangle')
    
    elif flaggeo == 15:
        sampling = np.zeros((imSize,imSize)) 
        for i in range(0,int(imSize/2)):
            sampling[int(imSize/2 + round(imSize/8))-i,int(imSize/8):int(imSize/2 + round(imSize/3)-i)] = 1.0
        print('...............Geometric Model > Off center trapezoid rectangle')            
    
    
    
    return sampling
