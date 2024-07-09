#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:42:05 2022

@author: pham
"""
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la
from scipy.special import kl_div

import time


#%% Computing error

def compute_error(V, WH):
    """
    Elementwise Kullback Leibler divergence

    Parameters
    ----------
    V : 2darray
        input data, left hand side of KL
    WH : 2d array
        right hand side of KL
    ind0 : boolean 2d array, optional
        table with True where V is not small, by default None
    ind1 : boolean 2d array, optional
        table with False where V is almost 0, by default None

    Returns
    -------
    float
        elementwise KL divergence

    """
    return np.sum(kl_div(V,WH))    

############################################################################
############################ PMF algorithm version Lee and Seung
    
def Lee_Seung_KL(V, Wini, Hini, nb_inner=10, NbIter=10000, epsilon=1e-8, tol=1e-7, legacy=False, verbose=False, print_it=100, delta=np.Inf):
    
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
    print_it: int
        if verbose is true, sets the number of iterations between each print.
        default: 100
    delta: float
        relative change between first and next inner iterations that should be reached to stop inner iterations dynamically.
        A good value empirically: 0.4
        default: np.Inf (no dynamic stopping)

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
    tic = time.perf_counter()

    if verbose:
        print("\n------Lee_Sung_KL running------")

    W = Wini.copy()
    H = Hini.copy()    
    WH = W.dot(H)
    crit = [compute_error(V, WH)]
    cnt = []
  
    if legacy:
        epsilon=0
     

    for k in range(NbIter):
        # FIXED H ESTIMATE W
        
        sumH = np.sum(H, axis = 1)[None,:]
        #inner_change_0 = 1
        #inner_change_l = np.Inf
        for l in range(nb_inner):
            #deltaW =  np.maximum(W *(((V/WH).dot(H.T))/sumH-1), epsilon-W)
            W = np.maximum(W * (((V/WH).dot(H.T))/sumH), epsilon)
            #W = W + deltaW
            WH = W.dot(H) 
            #if k>0:
                #if l==0:
                    #inner_change_0 = np.linalg.norm(deltaW)**2
                #else:
                    #inner_change_l = np.linalg.norm(deltaW)**2
                #if inner_change_l < delta*inner_change_0:
                    #break
        cnt.append(l+1)

        # FIXED W ESTIMATE H
        
        sumW = np.sum(W, axis = 0)[:, None]
        #inner_change_0 = 1
        #inner_change_l = np.Inf
        for l in range(nb_inner):    
            #deltaH = np.maximum(H * ((W.T.dot(V/WH))/sumW-1), epsilon-H)
            H = np.maximum(H * ((W.T.dot(V/WH))/sumW), epsilon)
            #H = H + deltaH
            WH = W.dot(H)
            #if k>0:
                #if l==0:
                    #inner_change_0 = np.linalg.norm(deltaH)**2
                #else:
                    #inner_change_l = np.linalg.norm(deltaH)**2
                #if inner_change_l < delta*inner_change_0:
                    #break
        cnt.append(l+1)   
 
        
        # compute the error 
        crit.append(compute_error(V, WH))
        toc.append(time.perf_counter()-tic)
        if verbose:
            if k%print_it==0:
                print("Loss at iteration {}: {}".format(k+1,crit[-1]))
        # Check if the error is small enough to stop the algorithm 
        if tol:
            if (crit[k] <= tol):
                if verbose:
                    print("Loss at iteration {}: {}".format(k+1,crit[-1]))
                return crit, W, H, tol, cnt
        
    if verbose:
        print("Loss at iteration {}: {}".format(k+1,crit[-1]))
    return crit, W, H, toc, cnt
    

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
    nb_inner = 50# nb of algo iterations
    NbIter = 3000
    
    # adding noise to the observed data
    sigma =  1e-6
    delta = 0.0

    # Printing
    verbose=True
     
    if sigma == 0:
        NbSeed = 1 # if without noise nb of noise = 0
    else:
        NbSeed = 1
    
    Error0 = np.zeros(NbSeed)
    Error1 = np.zeros(NbSeed)
    Error2 = np.zeros(NbSeed)     
    Error3 = np.zeros(NbSeed)
    
    NbIterStop0 = np.zeros(NbSeed)
    NbIterStop1 = np.zeros(NbSeed)
    NbIterStop2 = np.zeros(NbSeed)
    NbIterStop3 = np.zeros(NbSeed)

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
        crit0, W0, H0, toc0, cnt0 = Lee_Seung_KL(V, Wini, Hini, nb_inner=nb_inner,             
            epsilon=epsilon, verbose=verbose, NbIter=NbIter, delta=delta)
          
         
    
     ## ------------------Display objective functions
    fig = plt.figure(figsize=(6,3),tight_layout = {'pad': 0})    
    plt.semilogy(crit0, label = 'Lee and Seung', linewidth = 3)

    plt.title('Objective function values versus iterations', fontsize=14)# for different majorizing functions')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(r'$\log\left( || V - WH || \right)$', fontsize=14)
    plt.legend(fontsize = 14)
    plt.grid(True)
    results_path = 'Results/beta_divergence' 
    plt.savefig(results_path+'.eps', format='eps')       
    plt.legend(fontsize = 14)  
    
    plt.figure(figsize=(6,3),tight_layout = {'pad': 0})    
    plt.semilogy(toc0, crit0, label = 'Lee and Seung', linewidth = 3)
    plt.title('Objective function values versus time', fontsize=14)# for different majorizing functions')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(r'$\log\left( || V - WH || \right)$', fontsize=14)
    plt.legend(fontsize = 14)
    plt.grid(True)
    results_path = 'Results/beta_divergence' 
    plt.savefig(results_path+'.eps', format='eps')       
    plt.legend(fontsize = 14)  
      
# =============================================================================
#     plt.figure(figsize=(6,3),tight_layout = {'pad': 0})
#     k=10
#     plt.plot(np.convolve(cnt0, np.ones(k)/k, mode='valid')[::3])
#     plt.plot(np.convolve(cnt3, np.ones(k)/k, mode='valid')[::3])
#     plt.legend(["LeeSeung", "Proposed"])
# =============================================================================

    plt.show()

    
        