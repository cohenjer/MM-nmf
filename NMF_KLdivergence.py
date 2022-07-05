#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:04:24 2021

@author: pham
"""


## OLD FILE !!!

import numpy as np
#from numpy import linalg as la
from matplotlib import pyplot as plt
#from ismember import ismember
#from scipy.io import loadmat
from numpy import linalg as la

import time




############################################################################
############################ PMF algorithm version Lee and Seung
    
def NMF_KL_Lee_Seung(V, Vorig, W0, H0, NbIter, ind0, ind1):
    
    """
    The goal of this method is to factorize (approximately) the non-negative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in miminime [ V log (V/WH) - V + WH ] s.t. W, H >= 0
    
    
    References:  
        [1] Daniel D. Lee and H. Sebastian Seung.  Learning the parts of objects by non-negative matrix factorization.
        Nature, 1999
        [2]   Daniel D. Lee and H. Sebastian Seung. Algorithms for non-negative matrix factorization. In
        Advances in Neural Information Processing Systems. MIT Press, 2001   
    
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
        non-negative esimated matrix.

    """
    
    W = W0.copy()
    H = H0.copy()
    error_norm = np.prod(Vorig.shape)
    
    WH = W.dot(H)
    crit = [betadivergence(V, WH, ind0, ind1)]
    error = [la.norm(Vorig- WH)/error_norm]  
  
     
    
    for k in range(NbIter):
        # FIXED W ESTIMATE H
      
        aux_H = H/(np.sum(W, axis = 0)[:, None])  
        H =  H +  aux_H*(W.T.dot((V/WH)-1) )
        WH = W.dot(H)
        
        # FIXED H ESTIMATE W
  
        aux_W = W/(np.sum(H, axis = 1)[None,:])          
        W = W + aux_W*(((V/W.dot(H))-1).dot(H.T))
        WH = W.dot(H) 
        
        
        # compute the error 
        
        crit.append(betadivergence(V, WH, ind0, ind1))
        error.append(la.norm(Vorig- WH)/error_norm) 
        # Check if the error is smalle enough to stop the algorithm 
        if (error[k] <1e-7):
            
            return error, crit, W, H, k
            
        
    return error, crit, W, H, k  
    



############################################################################

def Fevotte_KL(V, Vorig, W0, H0, NbIter, ind0, ind1):

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

    Returns
    -------
    W and H such that V = WH
    crit: beta-divergence though iteration 
    
    Reference
    ----------
    C. Fevotte & J. Idier, "Algorithms for nonnegative matrix factorization
    with the beta-divergence ", Neural Compuation, 2011.

    """
    W = W0.copy()
    H = H0.copy()
     
    
    error_norm = np.prod(Vorig.shape)
    m, n = V.shape
    
    WH = W.dot(H)
 
    crit = [betadivergence(V, WH, ind0, ind1)]
    error = [la.norm(Vorig- WH)/error_norm]
        
    for k in range(1, NbIter):
        
        
        W = W * ((V/WH).dot(H.T))/np.repeat(np.sum(H, axis = 1)[None, :], m , axis = 0)
        scale = np.sum(W, axis = 0)
        WH = W.dot(H)
     
    
     
        H = H * (W.T.dot(V/WH))/np.repeat(scale[:, None],n, axis = 1)
        WH = W.dot(H)
     
    
     
        W = W * np.repeat(1/scale[None, :], m , axis = 0)
        H = H * np.repeat(scale[:, None], n, axis = 1)
               
        crit.append(betadivergence(V, WH, ind0, ind1))
        error.append(la.norm(Vorig- WH)/error_norm)

            
        if (error[k] <1e-7):
             
            return error, crit,  W, H, k 
    
 
    return error, crit, W, H, k  
    


############################################################################

def betadivergence(V, WH, ind0, ind1):
    
    
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
    
 
    return np.sum(V[ind1]* np.log(V[ind1]/(WH[ind1]+1e-10)) - V[ind1] + WH[ind1] ) + np.sum(WH[ind0])
     
   
    
 
############################################################################
############################ PROPOSED METHOD  
    
def NMF_proposed_KL(V, Vorig, W0, H0, sumW, sumH, NbIter, inter_iter, ind0, ind1, eps=1e-7):
    
    """
    The goal of this method is to factorize (approximately) the non-negative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in miminime [ V log (V/WH) - V + WH ] s.t. W, H >= 0
    
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
        non-negative esimated matrix.

    """
    
    W = W0.copy()
    H = H0.copy()
    WH = W.dot(H)
    inner_iter_total = 0 
    error_norm = np.prod(Vorig.shape)
    gamma = 1.9
    
    crit = [betadivergence(V, WH, ind0, ind1)]
    error = [la.norm(Vorig- WH)/error_norm]
    Vinv = 1/V
    eps     = eps*la.norm(V)
    
    for k in range(NbIter):
        # FIXED W ESTIMATE H
        
        sqrt_w = np.sqrt(np.sum(W, axis = 0))
        A = (W*(np.repeat(W.dot(sqrt_w)[:,None],W.shape[1],axis=1))).T
        aux_H = np.repeat(sqrt_w[:,None],H.shape[1], axis=1)/(A.dot(Vinv))         
        Wt = W.T
        
        for ih in range(inter_iter):
            H =  np.maximum(H + gamma*aux_H*(Wt.dot((V/WH)-1) ), 0)
            WH = W.dot(H)
        inner_iter_total +=inter_iter
          
        # FIXED H ESTIMATE W
        
        sqrt_h = np.sqrt(np.sum(H, axis = 1))
        B = np.repeat(sqrt_h.dot(H)[None,:], H.shape[0], axis=0)*H          
        aux_W = np.repeat(sqrt_h[None,:],W.shape[0], axis=0)/(Vinv.dot(B.T))            
        Ht = H.T
        for iw in range(inter_iter):       
            W = np.maximum(W + gamma*aux_W*(((V/WH)-1).dot(Ht) ), 0)
            WH = W.dot(H)
        inner_iter_total +=inter_iter
        
        # compute the error 
        
        crit.append(betadivergence(V, WH, ind0, ind1))
        error.append(la.norm(Vorig- WH)/error_norm)
        
        
        V_WH_1 = V/W.dot(H)-1
        dH,dW =  W.T.dot(V_WH_1), V_WH_1.dot(H.T)
        
 
        test  = la.norm(dH*(H>0) + np.minimum(dH,0)*(H==0), 2) +la.norm(dW*(W>0) + np.minimum(dW,0)*(W==0), 2) # eq. 21 p.2885 -> A RETRAVAILLER
        
        
        # Check if the error is smalle enough to stop the algorithm 
        if (test <= eps):
            return error, crit,  W, H, (k+inner_iter_total)
            
        
    return error, crit, W, H, (k+inner_iter_total)   
    





#####-------------------------------------------------------------
# NeNMF
# from https://www.academia.edu/7815546/NeNMF_An_Optimal_Gradient_Method_for_Nonnegative_Matrix_Factorization
#-------------------------------------------------------------
# created: # 2021 oct. 11
#-------------------------------------------------------------

def grad_H(V, W, Wt, H):
    return Wt.dot(V/(W.dot(H))-1)

def grad_W(V, W, H, Ht):
    return (V/(W.dot(H))-1).dot(Ht)

def OGM_H(V,W,H, L, nb_inner):
        # V≈WH, W≥O, H≥0
        # updates H        
        
        Y = H.copy()
        alpha     = 1
        Wt = W.T
 
        for ih in range(nb_inner):
            H_ = H.copy()
            alpha_ = alpha          
            H =  np.maximum(Y + L*grad_H(V, W, Wt, Y), 0)# projection entrywise on R+ of gradient step
            alpha = (1+np.sqrt(4*alpha**2+1))/2  # Nesterov momentum parameter           
            Y = H + ((alpha-1)/alpha_)*(H-H_)
            
        
        return H

def OGM_W(V,W,H, L, nb_inner):
        # V≈WH, W≥O, H≥0
        # updates W
        # eps: threshold for stopping criterion
        Ht = H.T
        alpha = 1
        Y = W.copy()
        for iw in range(nb_inner):
            W_ = W.copy()
            alpha_ = alpha 
            W = np.maximum(Y + L*grad_W(V,Y,H,Ht), 0)            
            alpha = (1+np.sqrt(4*alpha**2+1))/2  # Nesterov momentum parameter           
            Y = W + ((alpha-1)/alpha_)*(W-W_)
        return W
            

         
         
def NeNMF_KL(Vorig, V, W0, H0, ind0, ind1, eps=1e-7, nb_inner=10):
    W = W0.copy()
    H = H0.copy()
    error_norm = np.prod(Vorig.shape)
    WH = W.dot(H)
    
    crit = [betadivergence(V, WH, ind0, ind1)]
    error = [la.norm(Vorig- WH)/error_norm]
    
    test   = 1 # 
    inner_iter_total = 0
    eps     = eps*la.norm(V)
    while (test> eps):
        Lw = 1/la.norm(W.T.dot(W),2)
        H     = OGM_H(V, W, H, Lw, nb_inner)
        Lh = 1/la.norm(H.dot(H.T), 2)
        W     = OGM_W(V, W, H, Lh, nb_inner)
        inner_iter_total = inner_iter_total+2*nb_inner
        
        V_WH_1 = V/W.dot(H)-1
        dH,dW =  W.T.dot(V_WH_1), V_WH_1.dot(H.T)       
        WH = W.dot(H)
 
        test  = la.norm(dH*(H>0) + np.minimum(dH,0)*(H==0), 2) +la.norm(dW*(W>0) + np.minimum(dW,0)*(W==0), 2) # eq. 21 p.2885 -> A RETRAVAILLER
        
        error.append(la.norm(Vorig- WH)/error_norm)
        
        crit.append(betadivergence(V, WH, ind0, ind1))
            
    return error, crit, W, H, inner_iter_total


def NeNMF_optimMajo_KL(Vorig, V, W0, H, ind0, ind1, sumW, sumH, eps=1e-7, nb_inner=10):
    W = W0.copy()
    H = H0.copy()
    error_norm = np.prod(Vorig.shape)
    WH = W.dot(H)
    Vinv = 1/V
    crit = [betadivergence(V, WH, ind0, ind1)]
    error = [la.norm(Vorig- WH)/error_norm]
    
    test   = 1 # 
    inner_iter_total = 0
    eps     = eps*la.norm(V)
    while (test> eps) and (inner_iter_total<=10000):
        
        #----- Fixe W estimate H        
        sqrt_w = np.sqrt(np.sum(W, axis = 0))
        A = (W*(np.repeat(W.dot(sqrt_w)[:,None],W.shape[1],axis=1))).T
        Lw = np.repeat(sqrt_w[:,None],H.shape[1], axis=1)/(A.dot(Vinv))         
        H     = OGM_H(V, W, H, Lw, nb_inner)
        
        #----- Fixe H estimate W       
        sqrt_h = np.sqrt(np.sum(H, axis = 1))
        B = np.repeat(sqrt_h.dot(H)[None,:], H.shape[0], axis=0)*H          
        Lh = np.repeat(sqrt_h[None,:],W.shape[0], axis=0)/(Vinv.dot(B.T))          
        W     = OGM_W(V, W, H, Lh, nb_inner)
        inner_iter_total = inner_iter_total+2*nb_inner
        WH = W.dot(H)
        
        V_WH_1 = V/W.dot(H)-1
        dH,dW =  W.T.dot(V_WH_1), V_WH_1.dot(H.T)    
        test  = la.norm(dH*(H>0) + np.minimum(dH,0)*(H==0), 2) +la.norm(dW*(W>0) + np.minimum(dW,0)*(W==0), 2) # eq. 21 p.2885 -> A RETRAVAILLER
        
        error.append(la.norm(Vorig- WH)/error_norm)
        
        crit.append(betadivergence(V, WH, ind0, ind1))
            
    return error, crit, W, H, inner_iter_total



######################################################################
########### TEST

if __name__ == '__main__':
    
    plt.close('all')
    m = 70
    n = 60
    p = 40 
    
    Worig = np.random.rand(m, p)
    Horig = np.random.rand(p, n)

    Vorig =  Worig.dot(Horig)
    

    
    
    # adding noise to the observed data
    sigma = 0#.00001
    np.random.seed(0)

    N = sigma*np.random.rand(m,n)
    V = Vorig + N
    
    eps = 2.2204e-16
    ind0 = np.where(V <= eps)
    ind1 = np.where(V > eps)
    
    # Init
    Wini = np.random.rand(m, p) + 1 
    Hini = np.random.rand(p, n) + 1 
    
    
    WH = Worig.dot(Horig)

    sumH = (np.sum(WH, axis = 0))[None,:]   
    sumW = (np.sum(WH, axis = 1))[:, None]   
    
    
   
    # Parameters
    NbIter = 10000 # nb of algo iterations


 
 
    # Beta divergence 
    time_start0 = time.time()
    error0, crit0, W0, H0, nbiter0 = Fevotte_KL(V, Vorig, Wini, Hini, NbIter, ind0, ind1)
    time0 = time.time() - time_start0  
    crit0 = np.array(crit0)
    error0 = np.array(error0)
     
     
    inter_iter = 10
    time_start1 = time.time()
    #error1, crit1, W1, H1, nbiter1  = NeNMF_optimMajo_KL(Vorig, V, Wini, Hini, ind0, ind1, sumW, sumH) 
    error1, crit1, W1, H1, nbiter1  = NMF_proposed_KL(V, Vorig, Wini, Hini, sumW, sumH, NbIter, inter_iter, ind0, ind1)   

    time1 = time.time() - time_start1     
    crit1 = np.array(crit1)
    error1 = np.array(error1)  
    
    
    
    time_start2 = time.time()  
    error2, crit2, W2, H2, nbiter2  = NMF_KL_Lee_Seung(V, Vorig, Wini, Hini, NbIter, ind0, ind1)   
    time2 = time.time() - time_start2     
    crit2 = np.array(crit2)
    error2 = np.array(error2)  
    
    
    
    
     
    time_start3 = time.time()  
    error3, crit3, W3, H3, nbiter3  = NeNMF_KL(Vorig, V, Wini, Hini, ind0, ind1)
    time3 = time.time() - time_start3     
    crit3 = np.array(crit3)
    error3 = np.array(error3)  
     
    

    ## ------------------Display objective functions
    cst = 1e-12
 

    fig = plt.figure(figsize=(6,3),tight_layout = {'pad': 0})    
    plt.semilogy(error1 + cst, label = 'Pham et al', linewidth = 3)
    plt.semilogy(error2 + cst, label = 'Lee and Seung', linewidth = 3)
    plt.semilogy(error0 + cst,'--', label = 'Fevotte et al', linewidth = 3)
    plt.semilogy(error3 + cst,'--', label = 'NeNMF', linewidth = 3)
    plt.title('Reconstruction errors versus iterations', fontsize=14)# for different majorizing functions')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(r'$\log\left( || V - WH || \right)$', fontsize=14)
    plt.legend(fontsize = 14)
    plt.grid(True)
    results_path = 'Results/beta_divergence' 
    plt.savefig(results_path+'.eps', format='eps')       
    plt.legend(fontsize = 14)   
    
    
    fig = plt.figure(figsize=(6,3),tight_layout = {'pad': 0})    
    plt.semilogy(crit1 + cst, label = 'Pham et al', linewidth = 3)
    plt.semilogy(crit2 + cst, label = 'Lee and Seung', linewidth = 3)
    plt.semilogy(crit0 + cst,'--', label = 'Fevotte et al', linewidth = 3)
    plt.semilogy(crit3 + cst,'--', label = 'NeNMF', linewidth = 3)
    plt.title('Objective function values versus iterations', fontsize=14)# for different majorizing functions')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(r'$\log\left( || V - WH || \right)$', fontsize=14)
    plt.legend(fontsize = 14)
    plt.grid(True)
    results_path = 'Results/beta_divergence' 
    plt.savefig(results_path+'.eps', format='eps')       
    plt.legend(fontsize = 14)  
    
    print('Lee and Seung: Error = '+str(error2[-1]) + '; Crit = ' +str(crit2[-1]) + '; Elapsed time = '+str(time2)+ '\n')
    print('Fevotte et al: Error = '+str(error0[-1]) + '; Crit = ' + str(crit0[-1]) + '; Elapsed time = '+str(time0)+ '\n')
    print('Pham et al: Error = '+str(error1[-1]) + '; Crit = '+ str(crit1[-1]) + '; Elapsed time = '+str(time1)+ '\n')
    print('NeNMF: Error = '+str(error3[-1]) + '; Crit = '+ str(crit3[-1]) + '; Elapsed time = '+str(time3)+ '\n')

 