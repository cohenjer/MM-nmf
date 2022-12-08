#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 14:55:58 2021

@author: pham
"""

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
#from tempfile import TemporaryFile # save numpy arrays 
import time
#import tensorly as tl


# -----------------------------------
# Computing error efficiently
def compute_error(Vnorm_sq,WtW,H,WtV,error_norm):
    """
    This function computes \|V - WH \|_F /n/m with n,m the sizes of V. It does so without explicitely computing the norm but rather reusing previously computed cross products HHt and VHt. Vnorm_sq is the squared Frobenius norm of V.
    """
    return np.sqrt(np.abs(Vnorm_sq - 2*np.sum(WtV*H) +  np.sum(WtW*(H@H.T))))/error_norm

#------------------------------------
# PMF algorithm version Lee and Seung

def NMF_Lee_Seung(V, W, H0, NbIter, legacy=False, epsilon=1e-8, verbose=False, print_it=100, delta=np.Inf):
    
    """
    The goal of this method is to factorize (approximately) the non-negative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in miminime (1/2) || V - WH ||^2 s.t. W, H >= 0
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
    legacy: bool
        If True, implements the original update rule of Lee and Seung.
        If False, uses max( update, epsilon ) which ensures convergence with the BSUM framework and avoids zero-locking.
    epsilon: float
        if legacy is False, factors satisfy H > epsilon, W > epsilon instead of elementwise nonnegativity.
    tol: float
        stopping criterion, algorithm stops if error<tol.
    print_it: int
        if verbose is true, sets the number of iterations between each print.
        default: 100
    delta: float
        relative change between first and next inner iterations that should be reached to stop inner iterations dynamically.
        A good value empirically: 0.5
        default: np.Inf (no dynamic stopping)

    Returns
    -------
    error : darray
        vector that saves the error between Vorig with WH at each iteration.
    H : RxN array
        non-negative estimated matrix.
    W : MxR array
        non-negative esimated matrix.
    toc : darray
        vector containing the cummulative runtimes at each iteration

    """
 
    
    H = H0.copy()
    toc = [0] 
    tic = time.perf_counter()

    if verbose:
        print("\n--------- MU Lee and Sung running ----------")

    if legacy:
        epsilon=0

    # FIXED W ESTIMATE H      
    WtW = W.T@W
    WtV = W.T@V

    error_norm = np.prod(V.shape)
    Vnorm_sq = np.linalg.norm(V)**2
    error = [compute_error(Vnorm_sq,WtW,H,WtV,error_norm)]

    for k in range(NbIter):
        deltaH = np.maximum(H*(WtV/(WtW.dot(H)) - 1), epsilon-H)
        H = H + deltaH
        if k==0:
            inner_change_0 = np.linalg.norm(deltaH)**2
        else:
            inner_change_l = np.linalg.norm(deltaH)**2
            if inner_change_l < delta*inner_change_0:
                break


        # compute the error
        err = compute_error(Vnorm_sq,WtW,H,WtV,error_norm)
        error.append(err)
        toc.append(time.perf_counter() - tic)
        if verbose:
            if k%print_it==0:
                print("Error at iteration {}: {}".format(k+1,err))
        # check if the err is small enough to stop 

    return error, H, toc 

#------------------------------------
#  NMF algorithm proposed version

def NMF_proposed_Frobenius(V , W, H0, NbIter, epsilon=1e-8, verbose=False, use_LeeS=True, print_it=100, delta=np.Inf):
    
    """
    The goal of this method is to factorize (approximately) the non-negative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in miminime (1/2) || V - WH ||^2 s.t. W, H >= 0
    
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
    tol: float
        stopping criterion, algorithm stops if error<tol.
    use_LeeS: bool
        if True, the majorant is the elementwise maximum between the proposed majorant and Lee and Seung majorant.
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
    toc : darray
        vector containing the cummulative runtimes at each iteration

    """
    
    H = H0.copy()
    # TODO: Discuss
    if use_LeeS:
        gamma = 1#1.9
    else:
        gamma = 1.9
    toc = [0] 
    tic = time.perf_counter()

    if verbose:
        print("\n--------- MU proposed running ----------")
    
    # FIXED W ESTIMATE H
    A1 = W.T.dot(W)
    B1 = W.T@V
    sqrtB1 =np.sqrt(B1/np.sum(W,axis=0)[:,None])
    aux_H = sqrtB1/A1.dot(sqrtB1)        

    error_norm = np.prod(V.shape)
    Vnorm_sq = np.linalg.norm(V)**2
    error = [compute_error(Vnorm_sq,A1,H,B1,error_norm)]
    
    inner_change_0 = 1
    inner_change_l = np.Inf
    for k in range(NbIter):
        
        A1H = A1.dot(H)
        # TODO: HELP QUYEN DEBUG
        if use_LeeS:
            aux_H_used = np.maximum(aux_H, H/A1H)
        else:
            aux_H_used = aux_H
        deltaH =  np.maximum(gamma*aux_H_used*(B1 - A1H), epsilon-H)
        H = H + deltaH
        if k==0:
            inner_change_0 = np.linalg.norm(deltaH)**2
        else:
            inner_change_l = np.linalg.norm(deltaH)**2
            if inner_change_l < delta*inner_change_0:
                break

        err = compute_error(Vnorm_sq,A1,H,B1,error_norm)
        error.append(err)
        toc.append(time.perf_counter() - tic)
        if verbose:
            if k%print_it==0:
                print("Error at iteration {}: {}".format(k+1,err))
            
    return error, H, toc

################## Gradient descent method

def Grad_descent(V, W, H0, NbIter, epsilon=1e-8, verbose=False, print_it=100, delta=np.Inf):
    
    """"
    The goal of this method is to factorize (approximately) the non-negative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in miminime (1/2) || V - WH ||^2 s.t. W, H >= 0
    
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
    tol: float
        stopping criterion, algorithm stops if error<tol.
    
    Returns
    -------
    err : darray
        vector that saves the error between Vorig with WH at each iteration.
    H : RxN array
        non-negative estimated matrix.G
    W : MxR array
        non-negative esimated matrix.
    toc : darray
        vector containing the cummulative runtimes at each iteration

    """
    
    H = H0.copy()
    toc = [0] 
    tic = time.perf_counter()

    if verbose:
        print("\n--------- Gradient Descent running ----------")
     
    #inner_iter_total = 0 
    gamma = 1.9
    # FIXED W ESTIMATE H
    Aw = W.T.dot(W)      
    normAw = la.norm(Aw,2)
    WtV = W.T.dot(V)

    error_norm = np.prod(V.shape)
    Vnorm_sq = np.linalg.norm(V)**2
    error = [compute_error(Vnorm_sq,Aw,H,WtV,error_norm)]

    inner_change_0 = 1
    inner_change_l = np.Inf
    for k in range(NbIter):
        deltaH =  np.maximum((gamma/normAw)*(WtV - Aw@H),epsilon-H)
        H = H + deltaH
        if k==0:
            inner_change_0 = np.linalg.norm(deltaH)**2
        else:
            inner_change_l = np.linalg.norm(deltaH)**2
            if inner_change_l < delta*inner_change_0:
                break
        
        # compute the error 
        err = compute_error(Vnorm_sq,Aw,H,WtV,error_norm)
        error.append(err)
        toc.append(time.perf_counter()-tic)
        if verbose:
            if k%print_it==0:
                print("Error at iteration {}: {}".format(k+1,err))

    if verbose:
        print("Loss at iteration {}: {}".format(k+1,error[-1]))
    return error, H, toc




#####-------------------------------------------------------------
# NeNMF
# from https://www.academia.edu/7815546/NeNMF_An_Optimal_Gradient_Method_for_Nonnegative_Matrix_Factorization
#-------------------------------------------------------------
# created: # 2021 oct. 11
#-------------------------------------------------------------

 

def OGM_H(WtV, H, Aw, L, nb_inner, epsilon, delta, V, W, print_it=100, verbose=False, tic=time.perf_counter()):
    # V≈WH, W≥O, H≥0
    # updates H        
    error_norm = np.prod(V.shape)
    Vnorm_sq = np.linalg.norm(V)**2
    error = [compute_error(Vnorm_sq,Aw,H,WtV,error_norm)]
    toc = [0]
    #tic = time.perf_counter() 
    Y = H.copy()
    alpha = 1
    inner_change_0 = 1
    inner_change_l = np.Inf
    for ih in range(nb_inner):
        H_ = H.copy()
        alpha_ = alpha 
        deltaH = np.maximum(L*(WtV - Aw.dot(Y)),epsilon-Y) # projection entrywise on R+ of gradient step
        H = Y + deltaH
        alpha = (1+np.sqrt(4*alpha**2+1))/2  # Nesterov momentum parameter           
        Y = H + ((alpha-1)/alpha_)*(H-H_)
        if ih==0:
            inner_change_0 = np.linalg.norm(deltaH)**2
        else:
            inner_change_l = np.linalg.norm(deltaH)**2
            if inner_change_l < delta*inner_change_0:
                break
        
        # compute the error 
        err = compute_error(Vnorm_sq,Aw,H,WtV,error_norm)
        error.append(err)
        toc.append(time.perf_counter()-tic)
        if verbose:
            if ih%print_it==0:
                print("Error at iteration {}: {}".format(ih+1,err))
    if verbose:
        print("Loss at iteration {}: {}".format(ih,error[-1]))
    return H, error, toc

         
def NeNMF(V, W, H0, itermax=10000, epsilon=1e-8, verbose=False, print_it=100, delta=np.Inf):
    
    tic = time.perf_counter()
    H = H0.copy()
    if verbose:
        print("\n--------- NeNMF running ----------")

    Aw = W.T.dot(W)
    Lw = 1/la.norm(Aw,2)
    WtV = W.T@V
    H, error, toc = OGM_H(WtV, H, Aw, Lw, itermax, epsilon, delta, V, W, verbose=verbose, tic=tic)

    return error, H, toc


def NeNMF_optimMajo(V, W, H0, itermax = 10000, print_it=100, epsilon=1e-8, verbose=False, use_LeeS=True, delta=np.Inf):
    
    tic = time.perf_counter()
    H = H0.copy()
    if verbose:
        print("\n--------- MU extrapolated proposed running ----------")

    A1 = W.T.dot(W)
    B1 = W.T@V
    sqrtB1 =np.sqrt(B1/np.sum(W,axis=0)[:,None])
    Lw = sqrtB1/A1.dot(sqrtB1)        
    if use_LeeS:
        Lw = np.maximum(Lw, 1/la.norm(A1,2))
        
    H, error, toc = OGM_H(B1, H, A1, Lw, itermax, epsilon, delta, V, W, verbose=verbose, tic=tic)

    return error, H, toc
