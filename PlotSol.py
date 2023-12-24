#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:57:55 2020

@author: fran parra-rojas
"""

#def plotSol( Npix, frequency, antPosmeters, Nant, bm, DirtyMap, im_rec, method, MAP ):
def plotSol( Npix, frequency, antPosmeters, Nant, bm, DirtyMap, MAP, im_rec, method, order ):
    
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.pyplot as pt

    
    currcmap = cm.jet # color map (plot)
    
    xpos = antPosmeters[:,0]
    ypos = antPosmeters[:,1]
    
    wavelength = 3e8/frequency # wavelength in m
    
    import UVPlane as UVP
    uu,vv = UVP.UVPlane( Nant, antPosmeters, wavelength )
    

    f, axarr = pt.subplots(2, 3, figsize=(16,10));
    f.suptitle('Image reconstruction by %s and for Npix = %i and f = %.2f MHz' %(method,Npix,1.0e-6*frequency))
    axarr[0,0].scatter(xpos,ypos);
    axarr[0,0].set_xlabel("x (m)");
    axarr[0,0].set_ylabel("y (m)");
    axarr[0,0].grid(True)
    axarr[0,0].axis('equal')
    axarr[0,0].set_title('Antenna Layout Nant = %i' %Nant)

    axarr[0,1].scatter(uu,vv);
    axarr[0,1].set_xlabel("U ($\lambda$)");
    axarr[0,1].set_ylabel("V ($\lambda$)");
    axarr[0,1].grid(True)
    axarr[0,1].axis('equal')
    axarr[0,1].text(-5.1, 4, '$\lambda$ = %.2f m' %wavelength)
    axarr[0,1].set_title('UV-Plane, S(u,v)')
#
    axarr[0,2].imshow(10*np.log10(np.abs(bm)),cmap=currcmap);
    axarr[0,2].set_title('Dirty Beam, B(l,m)')
    axarr[0,2].invert_yaxis()
    axarr[0,2].set_xlabel("Npix");
    axarr[0,2].set_ylabel("Npix");
    axarr[0,2].set_xlim([0,Npix-1]);
    axarr[0,2].set_ylim([0,Npix-1]);
    axarr[0,2].text(5, 55, 'f = %.2f MHz' %(1e-6*frequency))

    axarr[1,0].imshow(DirtyMap,cmap=currcmap);
    axarr[1,0].set_xlabel("Npix")
    axarr[1,0].set_ylabel("Npix")
    axarr[1,0].set_xlim([Npix-Npix/2,Npix+Npix/2-1]);
    axarr[1,0].set_ylim([Npix+Npix/2-1,Npix-Npix/2]);
    axarr[1,0].set_title('Dirty Map, I(l,m)*B(l,m)')
    
    #print(np.shape(np.concatenate([im_rec[:,0],[0.0, 0.0]])))
    #print(np.shape(np.reshape(np.concatenate([im_rec[:,0],[0.0, 0.0]]))))

    if order <= 0:
        if method == 'disc':
            im_rec = im_rec[:,0]
            imreshape = np.reshape(im_rec,(Npix,Npix))
        elif method == 'dsvd':
            imreshape = np.reshape(im_rec,(Npix,Npix))
        elif method == 'tikh':
            imreshape = np.reshape(im_rec,(Npix,Npix))
    elif order == 1:
        if method == 'tgsvd':
            imreshape = np.reshape(im_rec,(Npix,Npix))
        elif method == 'tikh':
            imconcatenate = np.concatenate([im_rec[:,0],[0.0]])
            imreshape = np.reshape(imconcatenate,(Npix,Npix))
        elif method == 'disc':
            imconcatenate = np.concatenate([im_rec[:,0],[0.0]])
            imreshape = np.reshape(imconcatenate,(Npix,Npix))
        elif method == 'dsvd':
            imconcatenate = np.concatenate([im_rec[:,0],[0.0]])
            imreshape = np.reshape(imconcatenate,(Npix,Npix))
    elif order == 2:
        if method == 'tikh':
            imconcatenate = np.concatenate([im_rec[:,0],[0.0, 0.0]])
            imreshape = np.reshape(imconcatenate,(Npix,Npix))
        elif method == 'disc':
            imconcatenate = np.concatenate([im_rec[:,0],[0.0, 0.0]])
            imreshape = np.reshape(imconcatenate,(Npix,Npix))
        elif method == 'dsvd':
            imconcatenate = np.concatenate([im_rec[:,0],[0.0, 0.0]])
            imreshape = np.reshape(imconcatenate,(Npix,Npix))
   
    if method == 'lsqlin':
        imreshape = np.reshape(im_rec[:,0],(Npix,Npix))
    elif method == 'logmart' or method == 'kaczmarz':
        imreshape = np.reshape(im_rec,(Npix,Npix))
    elif method == 'maxent':
        imreshape = np.reshape(im_rec[:,0],(Npix,Npix))
    elif method == 'lanczos':
        imreshape = np.reshape(im_rec[:,9],(Npix,Npix))
    elif method == 'clean':
        imreshape = im_rec[:Npix,:Npix]
    elif method == 'mtsvd':
        im_rec = im_rec[:,5]
        imreshape = np.reshape(im_rec,(Npix,Npix))
    elif method == 'ttls':
        im_rec = im_rec[:,0]
        imreshape = np.reshape(im_rec,(Npix,Npix))
    elif method == 'tsvd':
        imreshape = np.reshape(im_rec,(Npix,Npix))
    elif method == 'tgsvd':
        imreshape = np.reshape(im_rec,(Npix,Npix))
        

    axarr[1,1].imshow(imreshape,cmap=currcmap);
    axarr[1,1].set_xlabel("Npix")
    axarr[1,1].set_ylabel("Npix")
    axarr[1,1].set_xlim([0,Npix-1]);
    axarr[1,1].set_ylim([Npix-1,0]);
    axarr[1,1].set_title('%s Solution' %method)

    axarr[1,2].imshow(MAP,cmap=currcmap);
    axarr[1,2].set_xlabel("Npix")
    axarr[1,2].set_ylabel("Npix")
    axarr[1,2].set_xlim([0,Npix-1]);
    axarr[1,2].set_ylim([Npix-1,0]);
    axarr[1,2].set_title('Map, I(l,m)')
    
    pt.show()

    pt.savefig("%s_%i_%.2f.png" %(method,Npix,frequency),format='png',dpi=300)