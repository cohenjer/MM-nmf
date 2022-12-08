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
        
    if ind0 or ind1:
        if not ind0:
            ind0 = np.zeros(V.shape,dtype=bool)
        if not ind1:
            ind1 = np.zeros(V.shape,dtype=bool)
        return np.sum(V[ind1]* np.log(V[ind1]/(WH[ind1]+1e-10)) - V[ind1] + WH[ind1] ) + np.sum(WH[ind0])
    return np.sum(kl_div(V,WH)) #V* np.log(V/WH) - V + WH)

# Stoppig criteria

#def Criteria_stopping(dH, H, dW, W):
    
    #return la.norm(dH*(H>0) + np.minimum(dH,0)*(H==0), 2) +la.norm(dW*(W>0) + np.minimum(dW,0)*(W==0), 2) # eq. 21 p.2885 -> A RETRAVAILLER 


############################################################################
############################ PMF algorithm version Lee and Seung
    
def Lee_Seung_KL(V,  Wini, Hini, ind0=None, ind1=None, nb_inner=10, NbIter=10000, epsilon=1e-8, tol=1e-7, legacy=False, verbose=False, print_it=100, delta=np.Inf):
    
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
    crit = [compute_error(V, WH, ind0, ind1)]
    cnt = []
  
    if legacy:
        epsilon=0
     

    for k in range(NbIter):
        
        
        # FIXED H ESTIMATE W
        
        sumH = (np.sum(H, axis = 1)[None,:]) 
        inner_change_0 = 1
        inner_change_l = np.Inf
        for l in range(nb_inner):
            deltaW =  np.maximum(W *(((V/WH).dot(H.T))/sumH-1), epsilon-W)
            W = W + deltaW
            WH = W.dot(H) 
            if l==0:
                inner_change_0 = np.linalg.norm(deltaW)**2
            else:
                inner_change_l = np.linalg.norm(deltaW)**2
            if inner_change_l < delta*inner_change_0:
                break
        cnt.append(l+1)

        # FIXED W ESTIMATE H
        
        sumW = np.sum(W, axis = 0)[:, None]
        inner_change_0 = 1
        inner_change_l = np.Inf
        for l in range(nb_inner):    
            deltaH = np.maximum(H * ((W.T.dot(V/WH))/sumW-1), epsilon-H)
            H = H + deltaH
            WH = W.dot(H)
            if l==0:
                inner_change_0 = np.linalg.norm(deltaH)**2
            else:
                inner_change_l = np.linalg.norm(deltaH)**2
            if inner_change_l < delta*inner_change_0:
                break
        cnt.append(l+1)   
 
        
        # compute the error 
        crit.append(compute_error(V, WH, ind0, ind1))
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
    


############################################################################ 
# Beta divergence method 
############################################################################

def Fevotte_KL(V, Wini, Hini, ind0=None, ind1=None, nb_inner=10, NbIter=10000, epsilon=1e-8, tol = 1e-7, legacy=False, verbose=False, print_it=100, delta=np.Inf):

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
    delta: float
        relative change between first and next inner iterations that should be reached to stop inner iterations dynamically.
        A good value empirically: 0.01
        default: np.Inf (no dynamic stopping)

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
    tic = time.perf_counter()

    if verbose:
        print("\n------Fevotte_Idier_KL running------")
    W = Wini.copy()
    H = Hini.copy()
    
    if legacy:
        epsilon=0
     
    m, n = V.shape   
    WH = W.dot(H)
 
    crit = [compute_error(V, WH, ind0, ind1)]
    cnt = []
     
    for k in range(NbIter):
        
        sumH = (np.sum(H, axis = 1)[None,:]) 
        inner_change_0 = 1
        inner_change_l = np.Inf
        for l in range(nb_inner):
            deltaW =  np.maximum(W *(((V/WH).dot(H.T))/sumH-1), epsilon-W)
            W = W + deltaW
            WH = W.dot(H) 
            if l==0:
                inner_change_0 = np.linalg.norm(deltaW)**2
            else:
                inner_change_l = np.linalg.norm(deltaW)**2
            if inner_change_l < delta*inner_change_0:
                break
        cnt.append(l)
              
        # FIXED W ESTIMATE H
        
        scale = np.sum(W, axis = 0)
        inner_change_0 = 1
        inner_change_l = np.Inf
        for l in range(nb_inner):    
            deltaH = np.maximum(H * ((W.T.dot(V/WH))/scale[:, None]-1), epsilon-H)
            H = H + deltaH
            WH = W.dot(H)
            if l==0:
                inner_change_0 = np.linalg.norm(deltaH)**2
            else:
                inner_change_l = np.linalg.norm(deltaH)**2
            if inner_change_l < delta*inner_change_0:
                break
        cnt.append(l)

        # Here is the main difference with Lee and Sung: normalization 
        # Should not change anything however...
        W = np.maximum(W / scale[None,:], epsilon)
        H = np.maximum(H * scale[:,None], epsilon)
               
        crit.append(compute_error(V, WH, ind0, ind1))
        toc.append(time.perf_counter()-tic)
        if verbose:
            if k%print_it==0:
                print("Loss at iteration {}: {}".format(k+1,crit[-1]))

        if tol: 
            if (crit[k] <= tol):
                if verbose:
                    print("Loss at iteration {}: {}".format(k+1,crit[-1]))
                return  crit, W, H, toc, cnt
    
    if verbose:
        print("Loss at iteration {}: {}".format(k+1,crit[-1]))
 
    return  crit, W, H, toc, cnt


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

def OGM_H(V,W,H, L, nb_inner, epsilon, delta=np.Inf, return_inner=True):
        # V≈WH, W≥O, H≥0
        # updates H        
        
        Y = H.copy()
        alpha     = 1
 
        inner_change_0 = 1
        inner_change_l = np.Inf
        for ih in range(nb_inner):
            H_ = H.copy()
            alpha_ = alpha          
            deltaH =  np.maximum(L*grad_H(V, W, Y), epsilon-Y)# projection entrywise on R+ of gradient step
            H = Y + deltaH
            alpha = (1+np.sqrt(4*alpha**2+1))/2  # Nesterov momentum parameter       
            Y = H +((alpha-1)/alpha_)*(H-H_)
            if ih==0:
                inner_change_0 = np.linalg.norm(deltaH)**2
            else:
                inner_change_l = np.linalg.norm(deltaH)**2
            if inner_change_l < delta*inner_change_0:
                break
                
        if return_inner:
            return H, ih+1
        return H

def OGM_W(V,W,H, L, nb_inner, epsilon, delta=np.Inf, return_inner=True):
        # V≈WH, W≥O, H≥0
        # updates W
        # eps: threshold for stopping criterion
        Y = W.copy()
        alpha     = 1
 
        inner_change_0 = 1
        inner_change_l = np.Inf
        for iw in range(nb_inner):
            W_ = W.copy()
            alpha_ = alpha          
            deltaW =  np.maximum(L*grad_W(V,Y,H), epsilon-Y)# projection entrywise on R+ of gradient step
            W = Y + deltaW
            alpha = (1+np.sqrt(4*alpha**2+1))/2  # Nesterov momentum parameter       
            Y = W +((alpha-1)/alpha_)*(W-W_)
            if iw==0:
                inner_change_0 = np.linalg.norm(deltaW)**2
            else:
                inner_change_l = np.linalg.norm(deltaW)**2
            if inner_change_l < delta*inner_change_0:
                break

        if return_inner:
            return W, iw+1
        return W

         
         
def NeNMF_KL(V, Wini, Hini, ind0=None, ind1=None, nb_inner=10, NbIter=10000, epsilon=1e-8, tol=1e-7, verbose=False, stepsize=None, print_it=100, delta=np.Inf):
    """
    TODO
    """
    toc = [0]
    tic = time.perf_counter()

    if verbose:
        print("\n------NeNMF_KL running------")
    W = Wini.copy()
    H = Hini.copy()
    #WH = W.dot(H)
    
    crit = [compute_error(V, W.dot(H), ind0, ind1)]
    cnt = []
     
     
    for  k in range(NbIter):
        # TODO: not the Lipschitz constant
        if not stepsize:
            Lw = 1/la.norm(W, 2)**2
        else:
            Lw = stepsize[0]
        H, cnt = OGM_H(V, W, H, Lw, nb_inner, epsilon, delta) 
        cnt.append(cnt)
        if not stepsize:
            Lh = 1/la.norm(H, 2)**2
        else:
            Lh = stepsize[1]
        W, cnt = OGM_W(V, W, H, Lh, nb_inner, epsilon, delta)
        cnt.append(cnt)
        
        crit.append(compute_error(V, W.dot(H), ind0, ind1))
        toc.append(time.perf_counter()-tic)
        if verbose:
            if k%print_it==0:
                print("Loss at iteration {}: {}".format(k+1,crit[-1]))
        if tol:
            if (crit[k] <= tol):
                if verbose:
                    print("Loss at iteration {}: {}".format(k+1,crit[-1]))
                return crit,  W, H, toc, cnt
            
    if verbose:
        print("Loss at iteration {}: {}".format(k+1,crit[-1]))
    return crit, W, H, toc, cnt
    


############################################################################
############################ PROPOSED METHOD  
############################################################################

    
def Proposed_KL(V, Wini, Hini, ind0=None, ind1=None, nb_inner=10,
                NbIter=10000, epsilon=1e-8, tol=1e-7, verbose=False, print_it=100, use_LeeS=True, delta=np.Inf):
    
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
    tic = time.perf_counter()
    if verbose:
        print("\n------Proposed_MU_KL running------")

    W = Wini.copy()
    H = Hini.copy()
    WH = W.dot(H)

    crit = [compute_error(V, WH, ind0, ind1)]
    cnt = []
    
    Vinv = 1/(V+1e-16)
    
    for k in range(NbIter):
        inner_change_0 = 1
        inner_change_l = np.Inf
        sum_H = np.sum(H, axis = 1)[None,:] 
        sum_H2= np.sum(H, axis = 0)[None,:]
        aux_W = 1/(Vinv.dot((H*sum_H2).T))
        for iw in range(nb_inner): 
            if use_LeeS:
                deltaW = np.maximum(np.maximum(aux_W, W/sum_H)*((V/WH).dot(H.T) - sum_H), epsilon-W)
            else:
                deltaW = np.maximum(aux_W*((V/WH).dot(H.T) - sum_H), epsilon-W)
            W = W + deltaW
            WH = W.dot(H)
            if iw==0:
                inner_change_0 = np.linalg.norm(deltaW)**2
            else:
                inner_change_l = np.linalg.norm(deltaW)**2
            if inner_change_l < delta*inner_change_0:
                break
           
        cnt.append(iw+1)
            
        # FIXED W ESTIMATE H  
        sum_W = np.sum(W, axis = 0)[:, None]
        sum_W2= np.sum(W, axis = 1)[:, None]
        inner_change_0 = 1
        inner_change_l = np.Inf
        
        aux_H =   1/((W*sum_W2).T.dot(Vinv))
        for ih in range(nb_inner):
            if use_LeeS:
                deltaH = np.maximum(np.maximum(aux_H, H/sum_W)*((W.T).dot(V/WH)- sum_W ), epsilon-H)
            else:
                deltaH = np.maximum(aux_H*((W.T).dot(V/WH)- sum_W ), epsilon-H)
            H = H + deltaH
            WH = W.dot(H)
            if ih==0:
                inner_change_0 = np.linalg.norm(deltaH)**2
            else:
                inner_change_l = np.linalg.norm(deltaH)**2
            if inner_change_l < delta*inner_change_0:
                break
            
        cnt.append(ih+1)

        # compute the error 
        crit.append(compute_error(V, WH, ind0, ind1))
        toc.append(time.perf_counter()-tic)
        if verbose:
            if k%print_it==0:
                print("Loss at iteration {}: {}".format(k+1,crit[-1]))
        # Check if the error is small enough to stop the algorithm 
        if tol:
            if (crit[k] <= tol):
                if verbose:
                    print("Loss at iteration {}: {}".format(k+1,crit[-1]))
                return crit,  W, H, toc, cnt
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
          
         
        #crit1, W1, H1, toc1  = Fevotte_KL(V, Wini, Hini, nb_inner=nb_inner, 
            #epsilon=epsilon, verbose=verbose, NbIter=NbIter, delta=0.4)
        #time1 = toc1[-1]  
        #crit1 = np.array(crit1)
        #Error1[s] = crit1[-1] 
        #NbIterStop1[s] = len(crit1)
        
        
        #stepsize=[1e-5,1e-5]
        ##stepsize=None
        ## TODO: remove from test
        #crit2, W2, H2, toc2  =NeNMF_KL(V, Wini, Hini, nb_inner=nb_inner, 
            #epsilon=epsilon, verbose=verbose, NbIter=NbIter, stepsize=stepsize, delta=0.01)   
        #time2 = toc2[-1]     
        #crit2 = np.array(crit2)
        #Error2[s] = crit2[-1] 
        #NbIterStop2[s] = len(crit2)
        
         
        crit3, W3, H3, toc3, cnt3  = Proposed_KL(V, Wini, Hini, nb_inner=nb_inner, 
            epsilon=epsilon, verbose=verbose, NbIter=NbIter, delta=delta, alpha_strategy="data_sum", print_it=100)
        
        
    
     ## ------------------Display objective functions
    fig = plt.figure(figsize=(6,3),tight_layout = {'pad': 0})    
    plt.semilogy(crit0, label = 'Lee and Seung', linewidth = 3)
    plt.semilogy(crit3, label = 'Pham et al', linewidth = 3)

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
    plt.semilogy(toc3, crit3, label = 'Pham et al', linewidth = 3)
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

    
        