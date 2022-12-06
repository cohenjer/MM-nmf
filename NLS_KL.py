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
        
    if ind0 or ind1:
        if not ind0:
            ind0 = np.zeros(V.shape,dtype=bool)
        if not ind1:
            ind1 = np.zeros(V.shape,dtype=bool)
        return np.sum(V[ind1]* np.log(V[ind1]/(WH[ind1]+1e-10)) - V[ind1] + WH[ind1] ) + np.sum(WH[ind0])
    return np.sum(V* np.log(V/WH) - V + WH)

# Stoppig criteria

#def Criteria_stopping(dH, H, dW, W):
    
    #return la.norm(dH*(H>0) + np.minimum(dH,0)*(H==0), 2) +la.norm(dW*(W>0) + np.minimum(dW,0)*(W==0), 2) # eq. 21 p.2885 -> A RETRAVAILLER 


############################################################################
############################ PMF algorithm version Lee and Seung
    
def Lee_Seung_KL(V,  W, Hini, ind0=None, ind1=None, NbIter=10000, epsilon=1e-8, legacy=False, verbose=False, print_it=100, delta=np.Inf):
    
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
    W : MxR array
        input mixing matrix with all entries are non-negative.
    H0 : RxN array
        matrix with all entries are non-negative.
    NbIter : int
        the maximum number of iterations.
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
    tic = time.time()

    if verbose:
        print("\n------Lee_Sung_KL running------")

    H = Hini.copy()    
    WH = W.dot(H)
    crit = [compute_error(V, WH, ind0, ind1)]
  
    if legacy:
        epsilon=0
    
    sumW = np.sum(W, axis = 0)[:, None]
    inner_change_0 = 1
    inner_change_l = np.Inf

    for k in range(NbIter):
        deltaH = np.maximum(H * ((W.T.dot(V/WH))/sumW-1), epsilon-H)
        H = H + deltaH
        WH = W.dot(H)
        if k==0:
            inner_change_0 = np.linalg.norm(deltaH)**2
        else:
            inner_change_l = np.linalg.norm(deltaH)**2
        if inner_change_l < delta*inner_change_0:
            break
 
        # compute the error 
        crit.append(compute_error(V, WH, ind0, ind1))
        toc.append(time.time()-tic)
        if verbose:
            if k%print_it==0:
                print("Loss at iteration {}: {}".format(k+1,crit[-1]))
        # Check if the error is small enough to stop the algorithm 
        
    if verbose:
        print("Loss at iteration {}: {}".format(k+1,crit[-1]))
    return crit, H, toc
    


############################################################################
############################ PROPOSED METHOD  
############################################################################

    
def Proposed_KL(V, W, Hini, ind0=None, ind1=None,
                NbIter=10000, epsilon=1e-8, verbose=False, print_it=100, use_LeeS=False, delta=np.Inf,
                equation="Quyen"):
    
    """
    The goal of this method is to factorize (approximately) the non-negative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in miminize [ V log (V/WH) - V + WH ] s.t. W, H >= 0
    
    Parameters
    ----------
    Vorig : MxN array 
        matrix with all entries are non-negative Vorig = W*H
    V : MxN array 
        observation matrix that is Vorig + B where B represents to the noise.
    W : MxR array
        input mixing matrix with all entries are non-negative.
    H0 : RxN array
        matrix with all entries are non-negative.
    NbIter : int
        the maximum number of iterations.
    delta: float
        relative change between first and next inner iterations that should be reached to stop inner iterations dynamically.
        A good value empirically: 0.01
        default: np.Inf (no dynamic stopping)
    alpha_strategy: string or float
        choose the strategy to fix alpha_n in the majorant computation. Three choices are implemented:
        - "data_sum": alpha_n is chosen as the sum of data rows (for H update) and data columns (for W update)
        - "factors_sum": alpha_n is chosen as the sum of factors columns
        - a float, e.g. alpha_strategy=1, to fix alpha to a specific constant.
 
    
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

    H = Hini.copy()
    WH = W.dot(H)
    
    # for Quyen's code
    Vinv = 1/V
    # for Jeremy's code
    VnormH = np.sum(np.abs(V),axis=0)
    crit = [compute_error(V, WH, ind0, ind1)]
    
    inner_change_0 = 1
    inner_change_l = np.Inf

    sum_W = np.sum(W, axis=0)[:,None]
    sum_W2= np.sum(W, axis=1)[:,None]
    aux_H =   1/((W*sum_W2).T.dot(Vinv))

    for k in range(NbIter):
        # FIXED W ESTIMATE H        
        if use_LeeS:
            deltaH = np.maximum(np.maximum(aux_H, H/sum_W)*((W.T).dot(V/WH)- sum_W ), epsilon-H)
        else:
            deltaH = np.maximum(aux_H*((W.T).dot(V/WH)- sum_W ), epsilon-H)
        H = H + deltaH
        WH = W.dot(H)
        if k==0:
            inner_change_0 = np.linalg.norm(deltaH)**2
        else:
            inner_change_l = np.linalg.norm(deltaH)**2
        if inner_change_l < delta*inner_change_0:
            break

        # compute the error 
        crit.append(compute_error(V, WH, ind0, ind1))
        toc.append(time.time()-tic)
        if verbose:
            if k%print_it==0:
                print("Loss at iteration {}: {}".format(k+1,crit[-1]))
        # Check if the error is small enough to stop the algorithm 

    if verbose:
        print("Loss at iteration {}: {}".format(k+1,crit[-1]))
    return crit, H, toc