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
    return np.sum(kl_div(V, WH))    

# Stoppig criteria

#def Criteria_stopping(dH, H, dW, W):
    
    #return la.norm(dH*(H>0) + np.minimum(dH,0)*(H==0), 2) +la.norm(dW*(W>0) + np.minimum(dW,0)*(W==0), 2) # eq. 21 p.2885 -> A RETRAVAILLER 


############################################################################
############################ PMF algorithm version Lee and Seung
    
def Lee_Seung_KL(V,  W, Hini, ind0=None, ind1=None, NbIter=10000, epsilon=1e-8, legacy=False, verbose=False, print_it=100, delta=np.inf):
    
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
        default: np.inf (no dynamic stopping)

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

    H = Hini.copy()    
    WH = W.dot(H)
    crit = [compute_error(V, WH, ind0, ind1)]
  
    if legacy:
        epsilon=0
    
    sumW = np.sum(W, axis = 0)[:, None]
    inner_change_0 = 1
    inner_change_l = np.inf

    for k in range(NbIter):
        Hnew = np.maximum(H * ((W.T.dot(V/WH))/sumW), epsilon)
        deltaH = Hnew - H
        H = Hnew
        WH = W.dot(H)
        if k>0:
            if k==1:
                inner_change_0 = np.linalg.norm(deltaH)**2
            else:
                inner_change_l = np.linalg.norm(deltaH)**2
            if inner_change_l < delta*inner_change_0:
                break
 
        # compute the error 
        crit.append(compute_error(V, WH, ind0, ind1))
        toc.append(time.perf_counter()-tic)
        if verbose:
            if k%print_it==0:
                print("Loss at iteration {}: {}".format(k+1,crit[-1]))
        # Check if the error is small enough to stop the algorithm 
        
    if verbose:
        print("Loss at iteration {}: {}".format(k+1,crit[-1]))
    return crit, H, toc
    

############################################################################
############################ Alternating Armijo GD METHOD  
############################################################################

def GD_KL(V, W, Hini, NbIter=10000, epsilon=1e-8, verbose=False, print_it=100, delta=np.inf):
    
    """
    The goal of this method is to factorize (approximately) the non-negative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in miminize [ V log (V/WH) - V + WH ] s.t. W, H >= 0

    The algorithm used is Alternating Projected Gradient Descent with naive Armijo backtracking to select the stepsize.
    
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
        default: np.inf (no dynamic stopping)
    
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
        print("\n------Proposed_MU_KL running------")

    H = Hini.copy()
    WH = W.dot(H)
    gamma = 1
    ARMIJO_CST = 0.01
    ARMIJO_DEC = 0.3
    
    crit = [compute_error(V, WH)]
    
    inner_change_0 = 1
    inner_change_l = np.inf

    WH = W.dot(H)

    for k in range(NbIter):

        gradH = (W.T).dot((WH-V)/WH)
        gradnorm = np.linalg.norm(gradH)**2
        for i in range(100):
            Hnew = np.maximum(H - gamma*gradH, epsilon)
            if crit[-1] - compute_error(V,W@Hnew) > gamma*ARMIJO_CST*gradnorm:
                deltaH = H - Hnew
                H = Hnew
                break
            gamma = gamma*ARMIJO_DEC
        WH = W.dot(H)
        gamma=gamma*10
        if k==1: #0?
            inner_change_0 = np.linalg.norm(deltaH)**2
        else:
            inner_change_l = np.linalg.norm(deltaH)**2
        if inner_change_l < delta*inner_change_0:
            break

        # compute the error 
        crit.append(compute_error(V, WH))
        toc.append(time.perf_counter()-tic)
        if verbose:
            if k%print_it==0:
                print("Loss at iteration {}: {}".format(k+1,crit[-1]))
        # Check if the error is small enough to stop the algorithm 

    if verbose:
        print("Loss at iteration {}: {}".format(k+1,crit[-1]))
    return crit, H, toc


############################################################################
############################ Other methods for review  
############################################################################


def ScalarNewton(V, W, Hini, NbIter=10000, epsilon=1e-8, verbose=False, print_it=100, delta=np.inf, method="CCD"):
    
    """
    The goal of this method is to factorize (approximately) the non-negative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in miminize [ V log (V/WH) - V + WH ] s.t. W, H >= 0
    
    Parameters
    ----------
    V : MxN array 
        observation matrix that is Vorig + B where B represents to the noise.
    W : MxR array
        input mixing matrix with all entries are non-negative.
    Hini : RxN array
        matrix with all entries are non-negative.
    NbIter : int
        the maximum number of iterations.
    delta: float
        relative change between first and next inner iterations that should be reached to stop inner iterations dynamically.
        A good value empirically: 0.01
        default: np.inf (no dynamic stopping)
    method: string
        "CCD": component-wise scalar second order, without monotonicity [Hsieh, Dhillon 2011] with $s=0$.
        "SN": adapted method from [Hien, Gillis 2021], with monotonicity guarantees.
    
    Returns
    -------
    err : darray
        vector that saves the error between Vorig with WH at each iteration.
    H : RxN array
        non-negative estimated matrix.

    """
    toc = [0]
    tic = time.perf_counter()
    if verbose:
        print("\n------Scalar Newton running------")

    H = Hini.copy()
    WH = W.dot(H)
    
    crit = [compute_error(V, WH)]
    
    inner_change_0 = 1
    inner_change_l = np.inf

    # Self concordant constant
    if method == "SN":
        chj = np.max((V > 0) / np.sqrt(V), axis=0)  # 1 by n

    sum_W = np.sum(W, axis=0)
    WH = W.dot(H)

    for k in range(NbIter):
      
        Hnew = np.copy(H)
        
        for q in range(H.shape[0]):
            
            # Update of a single components, similar to HALS
            grad = - (W[:, q]).dot(V/WH) + sum_W[q]
            hess = ((W[:, q]**2)).dot(V/(WH**2))  # elementwise 2d order derivative
            s = np.maximum(H[q, :] - grad/hess, epsilon)  # TODO epsilon write in article
            if method == "SN":
                # safe update
                d = s - H[q, :]
                lamb = chj*np.sqrt(hess)*np.abs(d)  # broadcasting check
                Hnew[q, :] = np.where((grad <= 0) + (lamb <= 0.683802), s, H[q, :] + (1/(1+lamb)) * d)
            else:
                Hnew[q, :] = s

            WH += np.outer(W[:, q], Hnew[q, :] - H[q, :])  # updated
               
        deltaH = Hnew - H
        H = Hnew
        if k == 1:
            inner_change_0 = np.linalg.norm(deltaH)**2
        else:
            inner_change_l = np.linalg.norm(deltaH)**2
        if inner_change_l < delta*inner_change_0:
            break

        # compute the error 
        crit.append(compute_error(V, WH))
        toc.append(time.perf_counter()-tic)
        if verbose:
            if k % print_it==0:
                print("Loss at iteration {}: {}".format(k+1,crit[-1]))
        # Check if the error is small enough to stop the algorithm 

    if verbose:
        print("Loss at iteration {}: {}".format(k+1,crit[-1]))
    return crit, H, toc

############################################################################
############################ PROPOSED METHOD  
############################################################################

    
def Proposed_KL(V, W, Hini, ind0=None, ind1=None, NbIter=10000, epsilon=1e-8, verbose=False, print_it=100, delta=np.inf, gamma=1.9, method="mSOM"):
    
    """
    The goal of this method is to factorize (approximately) the non-negative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in minimize [ V log (V/WH) - V + WH ] s.t. W, H >= 0
    
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
        default: np.inf (no dynamic stopping)
    alpha_strategy: string or float
        choose the strategy to fix alpha_n in the majorant computation. Three choices are implemented:
        - "data_sum": alpha_n is chosen as the sum of data rows (for H update) and data columns (for W update)
        - "factors_sum": alpha_n is chosen as the sum of factors columns
        - a float, e.g. alpha_strategy=1, to fix alpha to a specific constant.
    method: String
        Set method to "MUSOM" to use the choice of Lee and Seung u=H in the local majoration, without any approximation. The differences in performance come from the choice of gamma>1. Default: "mSOM".
    safe: boolean
        Use True to ensure the cost function decreases, checking error decrease at every iteration and replacing the update by MU.
    
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
        print(f"\n------{method} running------")

    H = Hini.copy()
    WH = W.dot(H)
    
    crit = [compute_error(V, WH, ind0, ind1)]
    
    inner_change_0 = 1
    inner_change_l = np.inf

    sum_W = np.sum(W, axis=0)[:,None]
    sum_W2 = np.sum(W, axis=1)[:,None]
    WH = W.dot(H)
    WW2 = (W*sum_W2).T

    for k in range(NbIter):
        # FIXED W ESTIMATE H        
            
        if method == "MUSOM":
            temp_grad = W.T@(V/WH)
            aux_H = gamma*H/temp_grad
            # Preconditionned proximal gradient step
            Hnew = np.maximum(H + aux_H*(temp_grad - sum_W), epsilon)
        elif method == "mSOM":
            aux_H = gamma*1/(WW2@(V/(WH**2)))
            # Preconditionned proximal gradient step
            mgrad = W.T@(V/WH) - sum_W
            Hnew = np.maximum(H + aux_H*mgrad, epsilon)
        
        deltaH = Hnew - H
        H = Hnew
        WH = W.dot(H)
        
        if k == 1:
            inner_change_0 = np.linalg.norm(deltaH)**2
        else:
            inner_change_l = np.linalg.norm(deltaH)**2
        if inner_change_l < delta*inner_change_0:
            break

        # compute the error 
        crit.append(compute_error(V, WH, ind0, ind1))
        toc.append(time.perf_counter()-tic)
        #if method != "MUSOM" and safe: 
            #maj_true_cost = crit[-2] - np.sum(deltaH*mgrad) + 1/2*np.sum(deltaH*((1/aux_H)*deltaH))
            #if maj_true_cost < crit[-1]:
                #print("********Warning: local upper bound is lower than the cost at local optimum*********")
        if verbose:
            if k % print_it == 0:
                print("Loss at iteration {}: {}".format(k+1,crit[-1]))
                #if safe and method != "MUSOM":
                    ## Computing cost and value of the minimum of majorant, to see if we have MM properties
                    ## For debug/cvg proof only?
                    ## Note: SUM requires exact minimization of the majorant. Are we doing that when
                    ## 1. we have delta>1 --> no I think
                    ## 2. we use projection on the NN orthant --> yes but with another majorant ? (prox)
                    #print("Computing cost and majorant cost at optimal local and at update")
                    #print(crit[-1])
                    #maj_cost = crit[-2] - 1/2*np.sum(mgrad*(aux_H*mgrad))
                    #maj_true_cost = crit[-2] - np.sum(deltaH*mgrad) + 1/2*np.sum(deltaH*((1/aux_H)*deltaH))
                    #print(maj_cost)
                    #print(maj_true_cost)
            
    if verbose:
        print("Loss at iteration {}: {}".format(k+1,crit[-1]))
    return crit, H, toc
# %%
