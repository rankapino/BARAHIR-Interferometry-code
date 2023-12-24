#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:10:14 2020

@author: fran parra-rojas
"""

def SelectBeam(antPos, wavelength, Nant, Npix, flagbeam):

    if flagbeam == 1:
        print('...............Beam Calculation >  Array Factor')
        bm, uu, vv  = ArrayFactor(antPos, wavelength, Nant, Npix )
    if flagbeam == 2:
        print('...............Beam Calculation >  Beam Factor')
        bm, uu, vv  = BeamFactor(antPos, wavelength, Nant, Npix )
    if flagbeam == 3:
        print('...............Beam Calculation >  Baseline Transform')
        bm, uu, vv  = BaselineTransform(antPos, wavelength, Nant, Npix )

    return bm, uu, vv

def BaselineTransform( antPos, wavelength, Nant, Npix ):
    

    import numpy as np 


    nbaselines = Nant*(Nant-1)/2    
    baselines1s = np.zeros((int(nbaselines),2))

    uu = np.zeros(int(2*nbaselines))
    vv = np.zeros(int(2*nbaselines))

    k = 0
    for i in list(range(Nant)):
        for j in list(range(i+1,Nant)):

            baselines1s[k,0] = (antPos[i][0]-antPos[j][0])/wavelength # wavelengths # x 
            baselines1s[k,1] = (antPos[i][1]-antPos[j][1])/wavelength # wavelengths # y 
            k = k+1

    # reflected baselines - units of wavelength 
    uu = np.append(baselines1s[:,0],-baselines1s[:,0]) # wavelengths 
    vv = np.append(baselines1s[:,1],-baselines1s[:,1]) # wavelengths 
    

    uu_meters = uu*wavelength # m
    vv_meters = vv*wavelength # m 


    baselinevectorarraywavelengths = baselines1s
    baselinevectorarraymeters = np.vstack((uu_meters,vv_meters)).T
 
    uvpixweight = 0.5  
    uvpointsperlambda = 1.0 
 
    obs_uv = np.zeros((Npix,Npix)) 
    
    int_u_max = 0 ; int_v_max = 0    
    int_u_min = 0 ; int_v_min = 0  
    
    
    for i in range(len(baselinevectorarraywavelengths)):
#        
        int_u = int(np.around(((baselinevectorarraywavelengths)[i,0])/uvpixweight*uvpointsperlambda)) 
       
        if int_u > int_u_max : int_u_max = int_u
        if int_u < int_u_min : int_u_min = int_u
##        
        int_v = int(np.around(((baselinevectorarraywavelengths)[i,1])/uvpixweight*uvpointsperlambda))
##        
        if int_v > int_v_max : int_v_max = int_v
        if int_v < int_v_min : int_v_min = int_v

        obs_uv[int_u,int_v] = 1.0
        obs_uv[-int_u,-int_v] = 1.0
##        
        obs_uv[0,0] = 1.0
        
    beamnorm = np.zeros((Npix,Npix)) 
    beam1d1dc = []
    
    beam1d1dc[:] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(obs_uv[:,:])))
    
    beam1d1d = np.abs(beam1d1dc)  
    beam2d = np.array(beam1d1d)
    
    beamnorm[:,:] = beam2d/np.max(beam2d)
    
    #SamplingFunction = np.abs(np.fft.fft2(beamnorm))
    SamplingFunction = obs_uv

    return beamnorm, uu, vv

def BeamFactor( antPos, wavelength, Nant, Npix ):
    
    import numpy as np 

    nbaselines = Nant*(Nant-1)/2    

    uu = np.zeros(int(2*nbaselines))
    vv = np.zeros(int(2*nbaselines))

    baselines1s = np.zeros((int(nbaselines),2))

    k = 0
    for i in list(range(Nant)):
        for j in list(range(i+1,Nant)):

            baselines1s[k,0] = (antPos[i][0]-antPos[j][0])/wavelength # wavelengths # x 
            baselines1s[k,1] = (antPos[i][1]-antPos[j][1])/wavelength # wavelengths # y 
            k = k+1

    # reflected baselines - units of wavelength 
    uu = np.append(baselines1s[:,0],-baselines1s[:,0]) # wavelengths 
    vv = np.append(baselines1s[:,1],-baselines1s[:,1]) # wavelengths 
    

    uu_meters = uu*wavelength # m
    vv_meters = vv*wavelength # m 


    baselinevectorarraywavelengths = baselines1s
    baselinevectorarraymeters = np.vstack((uu_meters,vv_meters)).T
    
    aax = baselinevectorarraymeters[:,0] 
    aay = baselinevectorarraymeters[:,1]
    
    lenaa = len(baselinevectorarraymeters)
    
    lmg = np.linspace(-1,1,Npix) 
    
    ll = lmg[np.indices((Npix,Npix))[1,:]] 
    mm = lmg[np.indices((Npix,Npix))[0,:]]
    
    v = np.zeros((Npix,Npix),'complex')
    for i in range(lenaa):
        v += np.exp(-2j*np.pi*(aax[i]*ll+aay[i]*mm)/wavelength)
    
    bm = np.zeros((Npix,Npix))
    bm[:] = np.abs(v)/np.abs(v).max()
    
    SamplingFunction = np.abs(np.fft.fft2(bm))
    
    return bm, uu, vv

def ArrayFactor( antPos, wavelength, Nant, Npix ):
    
    import numpy as np 

    nbaselines = Nant*(Nant-1)/2     

    uu = np.zeros(int(2*nbaselines))
    vv = np.zeros(int(2*nbaselines))

    baselines1s = np.zeros((int(nbaselines),2))

    k = 0
    for i in list(range(Nant)):
        for j in list(range(i+1,Nant)):

            baselines1s[k,0] = (antPos[i][0]-antPos[j][0])/wavelength # wavelengths # x 
            baselines1s[k,1] = (antPos[i][1]-antPos[j][1])/wavelength # wavelengths # y 
            k = k+1

    # reflected baselines - units of wavelength 
    uu = np.append(baselines1s[:,0],-baselines1s[:,0]) # wavelengths 
    vv = np.append(baselines1s[:,1],-baselines1s[:,1]) # wavelengths 
    
    aax = antPos[:,0] 
    aay = antPos[:,1]
    
    
    lenaa = len(antPos)
    
    lmg = np.linspace(-1,1,Npix) 
    
    ll = lmg[np.indices((Npix,Npix))[1,:]] 
    mm = lmg[np.indices((Npix,Npix))[0,:]]

    
    #v = np.zeros((Npix,Npix),'complex')
    v = np.zeros((Npix,Npix),dtype=np.csingle)
    for i in range(lenaa):
        v += np.exp(-2j*np.pi*(aax[i]*ll+aay[i]*mm)/wavelength)
    
    bm = np.zeros((Npix,Npix),dtype=np.single)
    bm[:] = np.abs(v)**2/np.abs(v).max()**2
    
    SamplingFunction = np.fft.fft2(bm)
    
    return bm, uu, vv
    
    
def BaselineTransform2( antPos, MAP, wavelength, Nant, Npix ):
    
    import numpy as np
    
    posarray = np.asarray(antPos)
    agua_x = posarray[:int(Nant),0]
    agua_y = posarray[:int(Nant),1]
    
    dx = agua_x - agua_x.min()
    dy = agua_y - agua_y.min()
    
    dmax = np.max([dx.max(),dy.max()])
    
    dnx = dx/dmax
    dny = dy/dmax
    
    width = len(MAP[0])
    bmax = wavelength * width/2
    
    print('Max. baselines =', bmax, 'km')
    
    x = bmax * dnx
    y = bmax * dny
    
    lx = np.zeros((int(Nant),int(Nant)))
    ly = np.zeros((int(Nant),int(Nant)))
    
    for i in range(int(Nant)):
        for j in range(int(Nant)):
            lx[i,j] = x[i] - x[j]
            ly[i,j] = y[i] - y[j]
            
    
    u_new = lx/wavelength
    v_new = ly/wavelength
    
    re_u = np.reshape(u_new,(len(x)**2), order='F')
    re_v = np.reshape(v_new,(len(y)**2), order='F')
    
    SamplingFunction = np.zeros((Npix,Npix))
    for k in range(len(re_u)):
        int_u = int((re_u)[k])
        int_v = int((re_v)[k])
        
        SamplingFunction[int_u,int_v]  = 1.0
    
    #SamplingFunction[0,0] = 1.0
    
    beam = np.zeros((Npix,Npix)) 
    beam1d1dc = []
    
    beam1d1dc[:] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(SamplingFunction[:,:])))
    
    beam2d = np.array(np.abs(beam1d1dc))
    
    beam[:,:] = beam2d/np.max(beam2d)
    
    return beam, re_u, re_v
    
    
    
