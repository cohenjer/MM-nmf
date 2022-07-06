#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:42:05 2022

@author: pham
"""
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la

import time


#%% Computing error

def compute_error(V, WH, ind0=None, ind1=None):
        
        
    """
    Method proposed in C. Fevotte & J. Idier, "Algorithms for nonnegative matrix factorization
    with the beta-divergence ", Neural Compuation, 2011.
    FOR THE KK CASE --> BETA = 1

    Parameters
    ----------
    V : non-negative matrix of size m x n  (data)
    WH : TYPE
        DESCRIPTION.
     
    Returns
    -----------
    beta-divergen

    """
        
    if ind0 or ind1:
        if not ind0:
            ind0 = np.zeros(V.shape,dtype=bool)
        if not ind1:
            ind1 = np.zeros(V.shape,dtype=bool)
        return np.sum(V[ind1]* np.log(V[ind1]/(WH[ind1]+1e-10)) - V[ind1] + WH[ind1] ) + np.sum(WH[ind0])
    return np.sum(V* np.log(V/WH) - V + WH)

#%% Stoppig criteria

def Criteria_stopping(dH, H, dW, W):
    
    return la.norm(dH*(H>0) + np.minimum(dH,0)*(H==0), 2) +la.norm(dW*(W>0) + np.minimum(dW,0)*(W==0), 2) # eq. 21 p.2885 -> A RETRAVAILLER 


############################################################################
############################ PMF algorithm version Lee and Seung
    
def Lee_Seung_KL(V,  Wini, Hini, ind0=None, ind1=None, nb_inner=10, NbIter=10000, epsilon=1e-8, tol=1e-7, legacy=False, verbose=False, print_it=100):
    
    """
    The goal of this method is to factorize (approximately) the non-negative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in miminize [ V log (V/WH) - V + WH ] s.t. W, H >= 0
    
    
    References:  
        [1] Daniel D. Lee and H. Sebastian Seung.  Learning the parts of objects by non-negative matrix factorization.
        Nature, 1999
        [2]   Daniel D. Lee and H. Sebastian Seung. Algorithms for non-negative matrix factorization. In
        Advances in Neural Information Processing Systems. MIT Press, 2001   
    
    Parameters
    ----------
    V : MxN array 
        observation matrix that is Vorig + B where B represents to the noise.
    W0 : MxR array
        matrix with all entries are non-negative.
    H0 : RxN array
        matrix with all entries are non-negative.
    NbIter : int
        the maximum number of iterations.
    NbIter_inner: int
        number of inner loops
    
    Returns
    -------
    err : darray
        vector that saves the error between Vorig with WH at each iteration.
    H : RxN array
        non-negative estimated matrix.
    W : MxR array
        non-negative estimated matrix.

    """
    toc = [0]
    tic = time.time()

    if verbose:
        print("\n------Lee_Sung_KL running------")

    W = Wini.copy()
    H = Hini.copy()    
    WH = W.dot(H)
    crit = [compute_error(V, WH, ind0, ind1)]
  
    if legacy:
        epsilon=0
     

    for k in range(NbIter):
        
        
        # FIXED H ESTIMATE W
        
        sumH = (np.sum(H, axis = 1)[None,:]) 
        for _ in range(nb_inner):
            #W =   W + ( W/sumH )*(((V/W.dot(H))-1).dot(H.T)) 
            W =  np.maximum(W *((V/WH).dot(H.T))/sumH, epsilon)
            WH = W.dot(H) 
              
        # FIXED W ESTIMATE H
        
        sumW = np.sum(W, axis = 0)[:, None]
        for _ in range(nb_inner):    
            H = np.maximum(H * (W.T.dot(V/WH))/sumW, epsilon)
            #H =   H +  (H/sumW)*(W.T.dot((V/WH)-1) ) 
            WH = W.dot(H)
            
 
        
        # compute the error 
        crit.append(compute_error(V, WH, ind0, ind1))
        toc.append(time.time()-tic)
        if verbose:
            if k%print_it==0:
                print("Loss at iteration {}: {}".format(k+1,crit[-1]))
        # Check if the error is small enough to stop the algorithm 
        if tol:
            if (crit[k] <= tol):
                return crit, W, H, tol 
        
    return crit, W, H, toc 
    


############################################################################ 
# Beta divergence method 
############################################################################

def Fevotte_KL(V, Wini, Hini, ind0=None, ind1=None, nb_inner=10, NbIter=10000, epsilon=1e-8, tol = 1e-7, legacy=False, verbose=False, print_it=100):

    """
    Method proposed in C. Fevotte & J. Idier, "Algorithms for nonnegative matrix factorization
    with the beta-divergence ", Neural Compuation, 2011.
    FOR THE KK CASE --> BETA = 1

    Parameters
    ----------
    V : non-negative matrix of size m x n (data)
    W0 : basis, non-negative matrix of size m x p 
    H0 : gains, non-negative matrix of size p x n.
    NbIter :number of iterations

    legacy: bool, default: False
        if True, update is thresholded so that W and H >= epsilon at all times.
        This ensures global convergence to a stationary point.

    Returns
    -------
    W and H such that V = WH
    crit: beta-divergence though iteration 
    
    Reference
    ----------
    C. Fevotte & J. Idier, "Algorithms for nonnegative matrix factorization
    with the beta-divergence ", Neural Compuation, 2011.

    """
    toc = [0]
    tic = time.time()

    if verbose:
        print("\n------Fevotte_Idier_KL running------")
    W = Wini.copy()
    H = Hini.copy()
    
    if legacy:
        epsilon=0
     
    m, n = V.shape   
    WH = W.dot(H)
 
    crit = [compute_error(V, WH, ind0, ind1)]
     
    for k in range(NbIter):
        
        sumH = np.repeat(np.sum(H, axis = 1)[None, :], m , axis = 0)
        for _ in range(nb_inner):
            W = np.maximum(W * ((V/WH).dot(H.T))/sumH, epsilon)
            WH = W.dot(H)
        
        # TODO: use broadcasting to remove repeats
        scale = np.sum(W, axis = 0)
        #sumW = np.repeat(scale[:, None],n, axis = 1)
        #sumW = scale[:, None]
        for _ in range(nb_inner):
            #H = np.maximum(H * (W.T.dot(V/WH))/sumW, epsilon)
            H = np.maximum(H * (W.T.dot(V/WH))/scale[:, None], epsilon)
            WH = W.dot(H)  

        # Here is the main difference with Lee and Sung: normalization 
        # Should not change anything however...
        #W = np.maximum (W * np.repeat(1/scale[None, :], m , axis = 0), epsilon)
        W = np.maximum (W / scale[None,:], epsilon)
        #H = np.maximum (H * sumW, epsilon)
        H = np.maximum (H * scale[:,None], epsilon)
               
        crit.append(compute_error(V, WH, ind0, ind1))
        toc.append(time.time()-tic)
        if verbose:
            if k%print_it==0:
                print("Loss at iteration {}: {}".format(k+1,crit[-1]))

        if tol: 
            if (crit[k] <= tol):
                return  crit,  W, H, toc
    
 
    return  crit, W, H, toc


#####-------------------------------------------------------------
# NeNMF
# from https://www.academia.edu/7815546/NeNMF_An_Optimal_Gradient_Method_for_Nonnegative_Matrix_Factorization
#-------------------------------------------------------------
# created: # 2021 oct. 11
#-------------------------------------------------------------

def grad_H(V, W, H):
    return (W.T).dot(V/(W.dot(H))-1)

def grad_W(V, W, H):
    return (V/(W.dot(H))-1).dot(H.T)

def OGM_H(V,W,H, L, nb_inner, epsilon):
        # V≈WH, W≥O, H≥0
        # updates H        
        
        Y = H.copy()
        alpha     = 1
 
        for ih in range(nb_inner):
            H_ = H.copy()
            alpha_ = alpha          
            H =  np.maximum(Y + L*grad_H(V, W, Y), epsilon)# projection entrywise on R+ of gradient step
            alpha = (1+np.sqrt(4*alpha**2+1))/2  # Nesterov momentum parameter           
            Y = H + ((alpha-1)/alpha_)*(H-H_)
            
        
        return H

def OGM_W(V,W,H, L, nb_inner, epsilon):
        # V≈WH, W≥O, H≥0
        # updates W
        # eps: threshold for stopping criterion
        alpha = 1
        Y = W.copy()
        for iw in range(nb_inner):
            W_ = W.copy()
            alpha_ = alpha 
            W = np.maximum(Y + L*grad_W(V,Y,H), epsilon)            
            alpha = (1+np.sqrt(4*alpha**2+1))/2  # Nesterov momentum parameter           
            Y = W + ((alpha-1)/alpha_)*(W-W_)
        return W
            

         
         
def NeNMF_KL(V, Wini, Hini, ind0=None, ind1=None, nb_inner=10, NbIter=10000, epsilon=1e-8, tol=1e-7, verbose=False, stepsize=None, print_it=100):
    """
    TODO
    """
    toc = [0]
    tic = time.time()

    if verbose:
        print("\n------NeNMF_KL running------")
    W = Wini.copy()
    H = Hini.copy()
    #WH = W.dot(H)
    
    crit = [compute_error(V, W.dot(H), ind0, ind1)]
     
     
    for  k in range(NbIter):
        # TODO: Is that the Lipschitz constant??
        if not stepsize:
            Lw = 1/la.norm(W, 2)**2
        else:
            Lw = stepsize[0]
        H     = OGM_H(V, W, H, Lw, nb_inner, epsilon) 
        if not stepsize:
            Lh = 1/la.norm(H, 2)**2
        else:
            Lh = stepsize[1]
        W     = OGM_W(V, W, H, Lh, nb_inner, epsilon)
        
        #V_WH_1 = V/W.dot(H)-1
        #dH,dW =  W.T.dot(V_WH_1), V_WH_1.dot(H.T)       
        #WH = W.dot(H)
 
        #test  = la.norm(dH*(H>0) + np.minimum(dH,0)*(H==0), 2) +la.norm(dW*(W>0) + np.minimum(dW,0)*(W==0), 2) # eq. 21 p.2885 -> A RETRAVAILLER
        
        #error.append(la.norm(Vorig- WH)/error_norm)
        
        crit.append(compute_error(V, W.dot(H), ind0, ind1))
        toc.append(time.time()-tic)
        if verbose:
            if k%print_it==0:
                print("Loss at iteration {}: {}".format(k+1,crit[-1]))
        if tol:
            if (crit[k] <= tol):
                return crit,  W, H, toc
            
    return crit, W, H, toc
    


############################################################################
############################ PROPOSED METHOD  
############################################################################

    
def Proposed_KL(V, Wini, Hini, sumH=None, sumW=None, ind0=None, ind1=None, nb_inner=10,
                NbIter=10000, epsilon=1e-8, tol=1e-7, verbose=False, print_it=100, use_LeeS=True):
    
    """
    The goal of this method is to factorize (approximately) the non-negative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in miminize [ V log (V/WH) - V + WH ] s.t. W, H >= 0
    
    Parameters
    ----------
    Vorig : MxN array 
        matrix with all entries are non-negative Vorig = W*H
    V : MxN array 
        observation matrix that is Vorig + B where B represents to the noise.
    W0 : MxR array
        matrix with all entries are non-negative.
    H0 : RxN array
        matrix with all entries are non-negative.
    NbIter : int
        the maximum number of iterations.
    NbIter_inner: int
        number of inner loops
    
    Returns
    -------
    err : darray
        vector that saves the error between Vorig with WH at each iteration.
    H : RxN array
        non-negative estimated matrix.
    W : MxR array
        non-negative estimated matrix.

    """
    toc = [0]
    tic = time.time()
    if verbose:
        print("\n------Proposed_MU_KL running------")

    W = Wini.copy()
    H = Hini.copy()
    WH = W.dot(H)

    # Precomputations
    if (not sumH) or (not sumW):
        # Using noisy data to compute sums
        sumH = (np.sum(V, axis = 0))[None,:]   
        sumW = (np.sum(V, axis = 1))[:, None]   
    
    crit = [compute_error(V, WH, ind0, ind1)]
     
    
    for k in range(NbIter):
        
        
        # FIXED H ESTIMATE W     
        sum_H = np.sum(H, axis = 1)[None,:]
        

        # TODO: repeat vs broadcasting? 
        aux_W = sumW/(np.sum(sum_H)*W.shape[0]*np.repeat(np.sqrt(sum_H),W.shape[0], axis=0))
        for iw in range(nb_inner):       
            if use_LeeS:
                W = np.maximum(W + np.maximum(aux_W, W/sum_H)*((V/WH).dot(H.T) - sum_H), epsilon)
            else:
                W = np.maximum(W + aux_W*((V/WH).dot(H.T) - sum_H), epsilon)
            WH = W.dot(H)
            
        # FIXED W ESTIMATE H        
        
        sum_W = np.sum(W, axis = 0)[:, None]          
        aux_H = sumH/(np.sum(sum_W)*H.shape[1]*np.repeat(np.sqrt(sum_W),H.shape[1], axis=1) )
        
        for ih in range(nb_inner):
            if use_LeeS:
                H = np.maximum(H + np.maximum(aux_H, H/sum_W)*((W.T).dot(V/WH)- sum_W ), epsilon)
            else:
                H = np.maximum(H + aux_H*((W.T).dot(V/WH)- sum_W ), epsilon)
 
            WH = W.dot(H)
          
        # compute the error 
        crit.append(compute_error(V, WH, ind0, ind1))
        toc.append(time.time()-tic)
        if verbose:
            if k%print_it==0:
                print("Loss at iteration {}: {}".format(k+1,crit[-1]))
        # Check if the error is small enough to stop the algorithm 
        if tol:
            if (crit[k] <= tol):
                return crit,  W, H, toc 
    return crit, W, H, toc    



######################################################################
########### TEST

if __name__ == '__main__':
    
    plt.close('all')
    m = 200
    n = 101
    p = 5
     
    Worig = np.random.rand(m, p) 
    Horig = np.random.rand(p, n) 
    Vorig =  Worig.dot(Horig) 
    
    # Init
    Wini = np.random.rand(m,p) + 1
    Hini = np.random.rand(p,n) + 1
    WH = Worig.dot(Horig)
   
    # Parameters
    nb_inner = 4# nb of algo iterations
    NbIter = 3000
    
    # adding noise to the observed data
    sigma = 0# 1e-6

    # Printing
    verbose=True 
     
    if sigma == 0:
        NbSeed = 1 # if without noise nb of noise = 0
    else:
        NbSeed = 5
    
    Error0 = np.zeros(NbSeed)
    Error1 = np.zeros(NbSeed)
    Error2 = np.zeros(NbSeed)     
    Error3 = np.zeros(NbSeed)
    
    NbIterStop0 = np.zeros(NbSeed)
    NbIterStop1 = np.zeros(NbSeed)
    NbIterStop2 = np.zeros(NbSeed)
    NbIterStop3 = np.zeros(NbSeed)

    # Why ??
    sumH = (np.sum(Vorig, axis = 0))[None,:]   
    sumW = (np.sum(Vorig, axis = 1))[:, None]   
        
    for  s in range(NbSeed): #[NbSeed-1]:#
        print('-------Noise with random seed =  ' +str(s)+'---------') 
        np.random.seed(s)

        N = sigma*np.random.rand(m,n)
        V = Vorig + N
        
        #eps = 2.2204e-16
        #ind0 = np.where(V <= eps)
        #ind1 = np.where(V > eps)

        epsilon = 1e-8
 
        # Beta divergence 
        time_start0 = time.time()
        crit0, W0, H0 = Lee_Seung_KL(V, Wini, Hini, nb_inner=nb_inner,             
            epsilon=epsilon, verbose=verbose, NbIter=NbIter)
        time0 = time.time() - time_start0  
        crit0 = np.array(crit0)
        Error0[s] = crit0[-1] 
        NbIterStop0[s] = len(crit0)
          
         
        time_start1 = time.time()
        crit1, W1, H1  = Fevotte_KL(V, Wini, Hini, nb_inner=nb_inner, 
            epsilon=epsilon, verbose=verbose, NbIter=NbIter)
        time1 = time.time() - time_start1     
        crit1 = np.array(crit1)
        Error1[s] = crit1[-1] 
        NbIterStop1[s] = len(crit1)
        
        
        time_start2 = time.time()  
        stepsize=[1e-5,1e-5]
        #stepsize=None
        crit2, W2, H2  =NeNMF_KL(V, Wini, Hini, nb_inner=nb_inner, 
            epsilon=epsilon, verbose=verbose, NbIter=NbIter, stepsize=stepsize)   
        time2 = time.time() - time_start2     
        crit2 = np.array(crit2)
        Error2[s] = crit2[-1] 
        NbIterStop2[s] = len(crit2)
        
         
        time_start3 = time.time()  
        crit3, W3, H3  = Proposed_KL(V, Wini, Hini, sumH=sumH, sumW=sumW, nb_inner=nb_inner, 
            epsilon=epsilon, verbose=verbose, NbIter=NbIter)
        time3 = time.time() - time_start3     
        crit3 = np.array(crit3)
        Error3[s] = crit3[-1] 
        NbIterStop3[s] = len(crit3)
        
        print('Lee and Seung: Crit = ' +str(crit0[-1]) + '; NbIter = '  + str(NbIterStop0[s]) + '; Elapsed time = '+str(time0)+ '\n')
        print('Fevotte et al: Crit = ' + str(crit1[-1]) + '; NbIter = '  + str(NbIterStop1[s]) + '; Elapsed time = '+str(time1)+ '\n')
        print('NeNMF: Crit = '+ str(crit2[-1]) + '; NbIter = '  + str(NbIterStop2[s]) + '; Elapsed time = '+str(time2)+ '\n')
        print('Pham et al: Crit = '+ str(crit3[-1]) + '; NbIter = '  + str(NbIterStop3[s]) + '; Elapsed time = '+str(time3)+ '\n')
        
        
    
     ## ------------------Display objective functions
    cst = 1e-12 
     
    fig = plt.figure(figsize=(6,3),tight_layout = {'pad': 0})    
    plt.semilogy(crit0 + cst, label = 'Lee and Seung', linewidth = 3)
    plt.semilogy(crit1 + cst,'--', label = 'Fevotte et al', linewidth = 3)
    plt.semilogy(crit2 + cst,'--', label = 'NeNMF', linewidth = 3)
    plt.semilogy(crit3 + cst, label = 'Pham et al', linewidth = 3)

    plt.title('Objective function values versus iterations', fontsize=14)# for different majorizing functions')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(r'$\log\left( || V - WH || \right)$', fontsize=14)
    plt.legend(fontsize = 14)
    plt.grid(True)
    results_path = 'Results/beta_divergence' 
    plt.savefig(results_path+'.eps', format='eps')       
    plt.legend(fontsize = 14)  
    
    
     


    
        