#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:14:14 2020

@author: fran parra-rojas
"""

def RegMethods( A, b, reg, flagreg, d, ks, gain, niter, fthresh ):
    
    if flagreg == 1:
        import numpy as np
        U, s, V = np.linalg.svd( A, full_matrices=False )
        V = V.T
        
        import get_l as GETL
        L = GETL.get_l( np.shape(A)[1], d )
        
        import mtsvd as MTSVD
        x_k, rho, eta = MTSVD.mtsvd( U, s, V, b, reg, L)
        method = 'MTSVD'
    
    elif flagreg == 2:
        import get_l as GETL
        L = GETL.get_l( np.shape(A)[1], d )
        
        import cgsvd as CGSVD
        U1, sm, X1, _, _ = CGSVD.cgsvd( A, L )
        
        import tgsvd as TGSVD
        x_k, rho, eta = TGSVD.tgsvd( U1, sm, X1, b, reg )
        method = 'TGSVD'
    
    elif flagreg == 3:
        import lsqlin as LSQLIN
        sol = LSQLIN.lsqlin( A, b )
        x_k = np.array(xsol['x'])
        method = 'LSQLIN'
    
    elif flagreg == 4:
        import numpy as np
        import logmart as LM
        x_k, chi2, i = LM.logmart( A, np.abs(b) )
        method = 'LOGMART'
    
    elif flagreg == 5:
        import kaczmarz as KM
        x_k, residual = KM.kaczmarz( A, b )
        method = 'Kaczmarz'
    
    elif flagreg == 6:
        import numpy as np
        U, s, V = np.linalg.svd( A, full_matrices=False )
        V = V.T
        
        import tsvd as TSVD
        x_k, rho, eta = TSVD.tsvd( U, s, V, b, reg )
        method = 'TSVD'
        
    elif flagreg == 7:
        TopB = np.column_stack(( A, b ))
        _ ,s , V = np.linalg.svd(TopB)
        V1 = V1.T
        
        import ttls as TTLS
        x_k, rho, eta = TTLS.ttls( V, reg , s )
        method = 'TTLS'
        
    elif flagreg == 8:
        import MaximumEntropy as MaxEnt
        x_k.squeeze(), rho, eta = MaxEnt.maxent( A, b, ks)
        method = 'MaxEnt'
        
    elif flagreg == 9:
        import tikhonov as TK
        x_k, rho, eta = TK.tikhonov( U, s, V, b, reg )
        method = 'Tikhonov'
    
    elif flagreg == 10:
        import numpy as np
        _, s, _ = np.linalg.svd(A)
        
        import lsqr_b as LSQRB
        x_k, rho, eta, F = LSQRB.lsqr_b( A, b, reg, 1, s):
        method = 'Didiag'
            
    elif flagreg == 11:
        import get_l as GETL
        L = GETL.get_l( np.shape(A)[1], d )
        
        import cgsvd as CGSVD
        U1, sm, _, V1, _ = CGSVD.cgsvd( A, L )
        
        import discrep as DS
        x_delta, x_k = DS.discrep( U1, sm, V1, b, reg )
        method = 'Morozov'
        
    elif flagreg == 12:
        import get_l as GETL
        L = GETL.get_l( np.shape(A)[1], d )
        
        import cgsvd as CGSVD
        U1, sm, _, V1, _ = CGSVD.cgsvd( A, L )

        import dsvd as DSVD
        x_k, rho, eta = DSVD.dsvd( U1, sm, V1, b, reg )
        method = 'DSVD'
        
    elif flagreg == 13:
        import CLEAN2 as CLEAN
        _, x_k = CLEAN.hogbom( b, A, gain, niter, fthresh )
        method = 'CLEAN'
        
        
    return x_k, rho, eta, method