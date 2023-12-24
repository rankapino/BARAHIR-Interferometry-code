#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:28:37 2020

@author: fran parra-rojas
"""

def RecImage( A, b, method, order, param, karray ):
    
    import sys
    import numpy as np
    
    if method == 'tikh':
        if param == 'gcv':
            if order < 0:
                U, s, V = np.linalg.svd( A, full_matrices = False )
                V = V.T
                s = np.reshape(s,(-1,1))
                
                import picard as PCD
                PCD.picard( U, s, b, d=0 )
            
            else:
                import get_l as GETL
                import cgsvd as CGSVD
                L = GETL.get_l( np.shape(A)[1], order )
                U, s, _, V, _ = CGSVD.cgsvd( A, L )
                
                import picard as PCD
                PCD.picard( U, s, b, d )
            
            b = b.flatten()
            bbar = np.reshape(b,(-1,1))
            
            import gcv as GCV
            reg_min, _, _ = GCV.gcv( U, s, bbar, 'tikh' )
        
        elif param == 'lcurve':
            if order < 0:
                U, s, V = np.linalg.svd( A, full_matrices = False )
                V = V.T
                s = np.reshape(s,(-1,1))
                
                import picard as PCD
                PCD.picard( U, s, b, d=0 ) 
                
            else:
                import get_l as GETL
                import cgsvd as CGSVD
                L = GETL.get_l( np.shape(A)[1], order )
                U, s, _, V, _ = CGSVD.cgsvd( A, L )
                
                import picard as PCD
                PCD.picard( U, s, b, d )
            
            b = b.flatten()
            bbar = np.reshape(b,(-1,1))
            
            import l_curve as LC
            reg_min, _, _, _ = LC.l_curve( U, s, bbar, 'tikh', L, V )
        
        elif param == 'ncp':
            if order < 0:
                U, s, V = np.linalg.svd( A, full_matrices = False )
                V = V.T
                s = np.reshape(s,(-1,1))
            else:
                import get_l as GETL
                import cgsvd as CGSVD
                L = GETL.get_l( np.shape(A)[1], order )
                U, s, _, V, _ = CGSVD.cgsvd( A, L ) 
            
            b = b.flatten()
            bbar = np.reshape(b,(-1,1))
            
            import ncp as NCP
            reg_min, _, _ = NCP.ncp( U, s, bbar, 'tikh' )

        
        elif param == 'qoc':
            if order < 0:
                U, s, V = np.linalg.svd( A, full_matrices = False )
                V = V.T
                s = np.reshape(s,(-1,1))
            else:
                import get_l as GETL
                import cgsvd as CGSVD
                L = GETL.get_l( np.shape(A)[1], order )
                U, s, _, V, _ = CGSVD.cgsvd( A, L ) 
            
            b = b.flatten()
            bbar = np.reshape(b,(-1,1))
            
            import quasiopt as QOC
            reg_min, _, _ = QOC.quasiopt( U, s, bbar, 'tikh' )

        import tikhonov as TK
        x_tik, _, _ = TK.tikhonov( U, s, V, bbar, reg_min[0] )
        
        
            
        else:
            print('Illegal Parameter Method')
            sys.exit()
    
    elif method == 'dsvd':
        if param == 'gcv':
            if order < 0:
                U, s, V = np.linalg.svd( A, full_matrices = False )
                V = V.T
                s = np.reshape(s,(-1,1))
            else:
                import get_l as GETL
                import cgsvd as CGSVD
                L = GETL.get_l( np.shape(A)[1], order )
                U, s, _, V, _ = CGSVD.cgsvd( A, L ) 
            
            b = b.flatten()
            bbar = np.reshape(b,(-1,1))
            
            import gcv as GCV
            reg_min, _, _ = GCV.gcv( U, s, bbar, 'dsvd' )
        
        elif param == 'lcurve':
            if order < 0:
                U, s, V = np.linalg.svd( A, full_matrices = False )
                V = V.T
                s = np.reshape(s,(-1,1))
            else:
                import get_l as GETL
                import cgsvd as CGSVD
                L = GETL.get_l( np.shape(A)[1], order )
                U, s, _, V, _ = CGSVD.cgsvd( A, L ) 
            
            b = b.flatten()
            bbar = np.reshape(b,(-1,1))
            
            import l_curve as LC
            reg_min, _, _, _ = LC.l_curve( U, s, bbar, 'dsvd', L, V )
        
        elif param == 'ncp':
            if order < 0:
                U, s, V = np.linalg.svd( A, full_matrices = False )
                V = V.T
                s = np.reshape(s,(-1,1))
            else:
                import get_l as GETL
                import cgsvd as CGSVD
                L = GETL.get_l( np.shape(A)[1], order )
                U, s, _, V, _ = CGSVD.cgsvd( A, L ) 
            
            b = b.flatten()
            bbar = np.reshape(b,(-1,1))
            
            import ncp as NCP
            reg_min, _, _ = NCP.ncp( U, s, bbar, 'dsvd' )

        elif param == 'qoc':
            if order < 0:
                U, s, V = np.linalg.svd( A, full_matrices = False )
                V = V.T
                s = np.reshape(s,(-1,1))
            else:
                import get_l as GETL
                import cgsvd as CGSVD
                L = GETL.get_l( np.shape(A)[1], order )
                U, s, _, V, _ = CGSVD.cgsvd( A, L ) 
            
            b = b.flatten()
            bbar = np.reshape(b,(-1,1))
            
            import quasiopt as QOC
            reg_min, _, _ = QOC.quasiopt( U, s, bbar, 'dsvd' )

        import dsvd as DSVD
        x_lambda, _, _ = DSVD.dsvd( U, s, V, bbar, reg_min)
    
    elif method == 'tsvd':
        if param == 'gcv':
            if order < 0:
                U, s, V = np.linalg.svd( A, full_matrices = False )
                V = V.T
                s = np.reshape(s,(-1,1))
            else:
                import get_l as GETL
                import cgsvd as CGSVD
                L = GETL.get_l( np.shape(A)[1], order )
                U, s, _, V, _ = CGSVD.cgsvd( A, L ) 
            
            b = b.flatten()
            bbar = np.reshape(b,(-1,1))
            
            import gcv as GCV
            reg_min, _, _ = GCV.gcv( U, s, bbar, 'tsvd' )
        
        elif param == 'lcurve':
            if order < 0:
                U, s, V = np.linalg.svd( A, full_matrices = False )
                V = V.T
                s = np.reshape(s,(-1,1))
            else:
                import get_l as GETL
                import cgsvd as CGSVD
                L = GETL.get_l( np.shape(A)[1], order )
                U, s, _, V, _ = CGSVD.cgsvd( A, L ) 
            
            b = b.flatten()
            bbar = np.reshape(b,(-1,1))
            
            import l_curve as LC
            reg_min, _, _, _ = LC.l_curve( U, s, bbar, 'tsvd', L, V )
        
        elif param == 'ncp':
            if order < 0:
                U, s, V = np.linalg.svd( A, full_matrices = False )
                V = V.T
                s = np.reshape(s,(-1,1))
            else:
                import get_l as GETL
                import cgsvd as CGSVD
                L = GETL.get_l( np.shape(A)[1], order )
                U, s, _, V, _ = CGSVD.cgsvd( A, L ) 
            
            b = b.flatten()
            bbar = np.reshape(b,(-1,1))
            
            import ncp as NCP
            reg_min, _, _ = NCP.ncp( U, s, bbar, 'tsvd' )

        
        elif param == 'qoc':
            if order < 0:
                U, s, V = np.linalg.svd( A, full_matrices = False )
                V = V.T
                s = np.reshape(s,(-1,1))
            else:
                import get_l as GETL
                import cgsvd as CGSVD
                L = GETL.get_l( np.shape(A)[1], order )
                U, s, _, V, _ = CGSVD.cgsvd( A, L ) 
            
            b = b.flatten()
            bbar = np.reshape(b,(-1,1))
            
            import quasiopt as QOC
            reg_min, _, _ = QOC.quasiopt( U, s, bbar, 'tsvd' )

        import tsvd as TSVD
        x_lambda, _, _ = TSVD.tsvd( U, s, V, bbar, reg_min )
        
    elif method == 'mtsvd':
        
        U, s, V = np.linalg.svd( A, full_matrices = False )
        V = V.T
        s = np.reshape(s,(-1,1))
        
        import get_l as GETL
        L = GETL.get_l( np.shape(A)[1], order )
        
        b = b.flatten()
        bbar = np.reshape(b,(-1,1))
        
        import mtsvd as MTSVD
        x_lambda, _, _ = MTSVD.mtsvd( U, s, V, bbar, karray, L )
    
    elif method == 'disc':
        
        if order < 0:
            U, s, V = np.linalg.svd( A, full_matrices = False )
            V = V.T
            s = np.reshape(s,(-1,1))
        else:
            import get_l as GETL
            import cgsvd as CGSVD
            L = GETL.get_l( np.shape(A)[1], order )
            U, s, _, V, _ = CGSVD.cgsvd( A, L ) 
            
        b = b.flatten()
        bbar = np.reshape(b,(-1,1))
        
        import discrep as DM
        x_lambda, _ = DM.discrep( U, s, V, bbar, karray )
    
    elif method == 'ttls':
        TopB = np.column_stack((A,b))
        U, s, V = np.linalg.svd( TopB )
        V = V.T
        s = np.reshape(s,(-1,1))
               
        import ttls as TTLS
        x_lambda, _, _ = TTLS.ttls( V, karray, s )
        
    elif method == 'lsqlin':
        
        import lsqlin  as LSQLIN
        xsol = LSQLIN.lsqlin( A, b ) 
        x_lambda = np.array(xsol['x'])
 
    elif method == 'logmart':
        
        import logmart as LM
        x_lambda, _, _ = LM.logmart( A, np.abs(b) )
        
    elif method == 'kaczmarz':
        
        import kaczmarz as KM
        x_lambda, _ = KM.kaczmarz( A, b )
    
    elif method == 'maxent':
        
        import MaximumEntropy as MEM
        x_lambda, _, _ = MEM.maxent( A, b, karray )
    
    elif method == 'lanczos':
        
        _, s, _ = np.linalg.svd( A, full_matrices = False )
        s = np.reshape(s,(-1,1))
        
        import lsqr_b as BL
        x_lambda, _, _, _ = BL.lsqr_b( A, b, karray, 1, s )
    
    else:
        print('Illegal Regularization Method')
        sys.exit()
    
    
    return x_lambda