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
def compute_error(Vnorm_sq,W,HHt,VHt,error_norm):
    """
    This function computes \|V - WH \|_F /n/m with n,m the sizes of V. It does so without explicitely computing the norm but rather reusing previously computed cross products HHt and VHt. Vnorm_sq is the squared Frobenius norm of V.
    """
    return np.sqrt(np.abs(Vnorm_sq - 2*np.sum(VHt*W) +  np.sum(HHt*(W.T@W))))/error_norm


#------------------------------------
# PMF algorithm version Lee and Seung

def NMF_Lee_Seung(V, W0, H0, NbIter, NbIter_inner, legacy=False, epsilon=1e-8, tol=1e-7, verbose=False, print_it=100, delta=np.Inf):
    
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
    W0 : MxR array
        matrix with all entries are non-negative.
    H0 : RxN array
        matrix with all entries are non-negative.
    NbIter : int
        the maximum number of iterations.
    NbIter_inner: int
        number of inner loops
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
 
    
    W = W0.copy()
    H = H0.copy()
    error_norm = np.prod(V.shape)
    error = [la.norm(V- W@H)/error_norm]
    Vnorm_sq = np.linalg.norm(V)**2
    toc = [0] 
    tic = time.time()
    cnt = []

    if verbose:
        print("\n--------- MU Lee and Sung running ----------")

    if legacy:
        epsilon=0

    for k in range(NbIter):
        
        # FIXED W ESTIMATE H      
        WtW = W.T@W
        WtV = W.T@V
        inner_change_0 = 1
        inner_change_l = np.Inf
        for j in range(NbIter_inner): 
            deltaH = np.maximum(H*(WtV/(WtW.dot(H)) - 1), epsilon-H)
            H = H + deltaH
            if j==0:
                inner_change_0 = np.linalg.norm(deltaH)**2
            else:
                inner_change_l = np.linalg.norm(deltaH)**2
            if inner_change_l < delta*inner_change_0:
                break
        cnt.append(j+1)

        # FIXED H ESTIMATE W
        VHt = V@H.T
        HHt = H@H.T
        inner_change_0 = 1
        inner_change_l = np.Inf
        for j in range(NbIter_inner):
            deltaW = np.maximum(W*(VHt/(W.dot(HHt))-1), epsilon-W)
            W = W + deltaW
            if j==0:
                inner_change_0 = np.linalg.norm(deltaW)**2
            else:
                inner_change_l = np.linalg.norm(deltaW)**2
            if inner_change_l < delta*inner_change_0:
                break
        cnt.append(j+1)

        # compute the error
        err = compute_error(Vnorm_sq,W,HHt,VHt,error_norm)
        error.append(err)
        toc.append(time.time() - tic)
        if verbose:
            if k%print_it==0:
                print("Error at iteration {}: {}".format(k+1,err))
        # check if the err is small enough to stop 
        if (error[-1] < tol):
            #if not legacy:
            #   # Putting zeroes where we thresholded with epsilon
            #   W[W==epsilon]=0 
            #   H[H==epsilon]=0
            return error, W, H, toc, cnt

    return error, W, H, toc, cnt

#------------------------------------
#  NMF algorithm proposed version

def NMF_proposed_Frobenius(V , W0, H0, NbIter, NbIter_inner, tol=1e-7, epsilon=1e-8, verbose=False, use_LeeS=True, print_it=100, delta=np.Inf, gamma=1.9):
    
    """
    The goal of this method is to factorize (approximately) the non-negative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in miminime (1/2) || V - WH ||^2 s.t. W, H >= 0
    
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
    gamma: float
        stepsize, default 1.9


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
    
    W = W0.copy()
    H = H0.copy()
    # TODO: Discuss
    if use_LeeS:
        gamma = 1
    error_norm = np.prod(V.shape)
    error = [la.norm(V-W.dot(H))/error_norm]
    Vnorm_sq = np.linalg.norm(V)**2
    toc = [0] 
    tic = time.time()
    cnt = []

    if verbose:
        print("\n--------- MU proposed running ----------")

    for k in range(NbIter):
        
        # FIXED W ESTIMATE H
        A1 = W.T.dot(W)
        B1 = W.T@V
        sqrtB1 =np.sqrt(B1)
        aux_H = gamma*sqrtB1/A1.dot(sqrtB1)        
        inner_change_0 = 1
        inner_change_l = np.Inf
        for ih in range(NbIter_inner):
            A1H = A1.dot(H)
            # TODO: HELP QUYEN DEBUG
            if use_LeeS:
                aux_H_used = np.maximum(aux_H, H/A1H)
            else:
                aux_H_used = aux_H
            deltaH =  np.maximum(aux_H_used*(B1 - A1H), epsilon-H)
            H = H + deltaH
            if ih==0:
                inner_change_0 = np.linalg.norm(deltaH)**2
            else:
                inner_change_l = np.linalg.norm(deltaH)**2
            if inner_change_l < delta*inner_change_0:
                break
        cnt.append(ih+1)

        # FIXED H ESTIMATE W
        A2 = H.dot(H.T)
        B2 = V@H.T
        sqrtB2 = np.sqrt(B2)
        aux_W = gamma*sqrtB2/sqrtB2.dot(A2)    
        inner_change_0 = 1
        inner_change_l = np.Inf
        for iw in range(NbIter_inner):
            WA2 = W.dot(A2)
            if use_LeeS:
                aux_W_used = np.maximum(aux_W, W/WA2)
            else:
                aux_W_used = aux_W
            deltaW = np.maximum(aux_W_used*(B2 - WA2), epsilon-W)
            W = W + deltaW
            if iw==0:
                inner_change_0 = np.linalg.norm(deltaW)**2
            else:
                inner_change_l = np.linalg.norm(deltaW)**2
            if inner_change_l < delta*inner_change_0:
                break
        cnt.append(iw+1)
               
        err = compute_error(Vnorm_sq,W,A2,B2,error_norm)
        error.append(err)
        toc.append(time.time() - tic)
        if verbose:
            if k%print_it==0:
                print("Error at iteration {}: {}".format(k+1,err))
        # Check if the error is smalle enough to stop the algorithm 
        if (error[-1] <tol):            
            print('algo stop at iteration =  '+str(k))
            return error, W, H, toc, cnt
            
        
    return error, W, H, toc, cnt


################## Gradient descent method

def Grad_descent(V , W0, H0, NbIter, NbIter_inner, tol=1e-7, epsilon=1e-8, verbose=False, print_it=100, delta=np.Inf, gamma=1.9):
    
    """"
    The goal of this method is to factorize (approximately) the non-negative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in miminime (1/2) || V - WH ||^2 s.t. W, H >= 0
    
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
    tol: float
        stopping criterion, algorithm stops if error<tol.
    gamma: float
        stepsize (multiplied by inverse of Lipschitz constant), default 1.9 (aggressive)
    
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
    
    
    
    W = W0.copy()
    H = H0.copy()
    error_norm = np.prod(V.shape)
    error = [la.norm(V- W.dot(H))/error_norm]
    Vnorm_sq = np.linalg.norm(V)**2
    toc = [0] 
    tic = time.time()
    cnt = []

    if verbose:
        print("\n--------- Gradient Descent running ----------")
     
    #inner_iter_total = 0 

    for k in range(NbIter):
        # FIXED W ESTIMATE H
        Aw = W.T.dot(W)      
        normAw = la.norm(Aw,2)
        WtV = W.T.dot(V)
        inner_change_0 = 1
        inner_change_l = np.Inf
        for ih in range(NbIter_inner):          
            deltaH =  np.maximum((gamma/normAw)*(WtV - Aw@H),epsilon-H)
            H = H + deltaH
            if ih==0:
                inner_change_0 = np.linalg.norm(deltaH)**2
            else:
                inner_change_l = np.linalg.norm(deltaH)**2
            if inner_change_l < delta*inner_change_0:
                break
        cnt.append(ih+1)
          
        # FIXED H ESTIMATE W
        Ah = H.dot(H.T)
        normAh = la.norm(Ah,2)
        VHt = V.dot(H.T)
        inner_change_0 = 1
        inner_change_l = np.Inf
        for iw in range(NbIter_inner):       
            deltaW = np.maximum((1.9/normAh)*(VHt - W@Ah),epsilon-W)
            W = W + deltaW
            if iw==0:
                inner_change_0 = np.linalg.norm(deltaW)**2
            else:
                inner_change_l = np.linalg.norm(deltaW)**2
            if inner_change_l < delta*inner_change_0:
                break
        cnt.append(iw+1)
        
        # compute the error 
        err = compute_error(Vnorm_sq,W,Ah,VHt,error_norm)
        error.append(err)
        toc.append(time.time()-tic)
        if verbose:
            if k%print_it==0:
                print("Error at iteration {}: {}".format(k+1,err))

        # Check if the error is small enough to stop the algorithm 
        if (error[-1] <tol):
        
            return error, W, H, toc, cnt
            
        
    if verbose:
        print("Loss at iteration {}: {}".format(k+1,error[-1]))
    return error, W, H, toc, cnt




#####-------------------------------------------------------------
# NeNMF
# from https://www.academia.edu/7815546/NeNMF_An_Optimal_Gradient_Method_for_Nonnegative_Matrix_Factorization
#-------------------------------------------------------------
# created: # 2021 oct. 11
#-------------------------------------------------------------

 

def OGM_H(WtV, H, Aw, L, nb_inner, epsilon, delta, return_inner=False):
        # V≈WH, W≥O, H≥0
        # updates H        
        
        Y = H.copy()
        alpha     = 1
        #Aw = W.T.dot(W)
        #L = la.norm(Aw,2)
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
        if return_inner:
            return H, ih+1
        return H

def OGM_W(VHt,W, Ah, L, nb_inner, epsilon, delta, return_inner=False):
        # V≈WH, W≥O, H≥0
        # updates W
        # eps: threshold for stopping criterion
        
        #Ah = H.dot(H.T)
        #L = la.norm(Ah,2)
        alpha = 1
        Y = W.copy()
        inner_change_0 = 1
        inner_change_l = np.Inf
        for iw in range(nb_inner):
            W_ = W.copy()
            alpha_ = alpha 
            deltaW = np.maximum(L*(VHt - Y.dot(Ah)),epsilon-Y)
            W = Y + deltaW
            alpha = (1+np.sqrt(4*alpha**2+1))/2  # Nesterov momentum parameter          
            Y = W + ((alpha-1)/alpha_)*(W-W_)
            if iw==0:
                inner_change_0 = np.linalg.norm(deltaW)**2
            else:
                inner_change_l = np.linalg.norm(deltaW)**2
            if inner_change_l < delta*inner_change_0:
                break
        if return_inner:
            return W, iw+1
        return W
            
        
         
def NeNMF(V, W0, H0, tol=1e-7, nb_inner=10, itermax=10000, epsilon=1e-8, verbose=False, print_it=100, delta=np.Inf):
    W = W0.copy()
    H = H0.copy()
    error_norm = np.prod(V.shape)
    error = [la.norm(V- W.dot(H))/error_norm]
    Vnorm_sq = np.linalg.norm(V)**2
    toc = [0] 
    tic = time.time()
    cnt = []

    if verbose:
        print("\n--------- NeNMF running ----------")

    it = 0
    while (error[-1]> tol) and (it<itermax): 

        Aw = W.T.dot(W)
        Lw = 1/la.norm(Aw,2)
        WtV = W.T@V
        H, cnt_inner = OGM_H(WtV, H, Aw, Lw, nb_inner,epsilon, delta, return_inner=True)
        cnt.append(cnt_inner)

        Ah = H.dot(H.T)
        Lh = 1/la.norm(Ah,2)
        VHt = V@H.T
        W, cnt_inner = OGM_W(VHt, W, Ah, Lh, nb_inner,epsilon, delta, return_inner=True)
        cnt.append(cnt_inner)

        err = compute_error(Vnorm_sq,W,Ah,VHt,error_norm)
        error.append(err)
        toc.append(time.time()-tic)
        if verbose:
            if it%print_it==0:
                print("Error at iteration {}: {}".format(it+1,err))
        it+=1
        
    if verbose:
        print("Loss at iteration {}: {}".format(it,error[-1]))
    return error, W, H, toc, cnt


def NeNMF_optimMajo(V, W0, H0, tol=1e-7, nb_inner=10, itermax = 10000, print_it=100, epsilon=1e-8, verbose=False, use_best=False, delta=np.Inf, gamma=1):
    W = W0.copy()
    H = H0.copy()
    error_norm = np.prod(V.shape)
    error = [la.norm(V- W.dot(H))/error_norm]
    Vnorm_sq = np.linalg.norm(V)**2
    toc = [0] 
    tic = time.time()
    cnt_inner = []
    if verbose:
        print("\n--------- MU extrapolated proposed running ----------")
    it = 0
    while error[-1]>tol and it < itermax:
        
        #----fixed w estimate H
        
        A1 = W.T.dot(W)
        B1 = W.T@V
        sqrtB1 =np.sqrt(B1)
        Lw = gamma*sqrtB1/A1.dot(sqrtB1)        
        if use_best:
            Lw = np.maximum(Lw, 1/la.norm(A1,2))
        
        #Lw = 1/la.norm(Aw,2)
        H, out_cnt = OGM_H(B1, H, A1, Lw, nb_inner, epsilon, delta, return_inner=True)
        cnt_inner.append(out_cnt)
        
        # fixed h estimate w
        
        A2 = H.dot(H.T)
        B2 = V@H.T
        sqrtB2 = np.sqrt(B2)
        Lh = sqrtB2/sqrtB2.dot(A2)    
        if use_best:
            Lh = np.maximum(Lh,1/la.norm(A2,2))

        W, out_cnt = OGM_W(B2, W, A2, Lh, nb_inner, epsilon, delta, return_inner=True)
        cnt_inner.append(out_cnt)
        err = compute_error(Vnorm_sq,W,A2,B2,error_norm)
        error.append(err)
        toc.append(time.time()-tic)
        if verbose:
            if it%print_it==0:
                print("Error at iteration {}: {}".format(it+1,err))
        it += 1
    
    if verbose:
        print("Loss at iteration {}: {}".format(it,error[-1]))
    return error, W, H, toc, cnt_inner




###################################################################
# TEST ALGORITHMS



if __name__ == '__main__':
    
    plt.close('all')
    # Fixe the matrix sizes
    
    rV = 70
    cV = 60
    cW = 30
    
    # Max number of iterations
    NbIter = 10000
    
    # Number of bruits 
    NbSeed = 1 
    
    
    # Fixed the signal 
    np.random.seed(NbSeed)
    Worig = np.random.rand(rV, cW) #sparse.random(rV, cW, density=0.25).toarray()  #
    
    np.random.seed(NbSeed + 1)
    Horig = np.random.rand(cW, cV)  
    
    # indw0 = np.random.randint(0,cW,int(cW/2))
    # indh0 = np.setdiff1d(range(cW),indw0)
    # Worig[0,indw0] = 0
    # Horig[indh0, 0] = 0
    
    Vorig = Worig.dot(Horig)
    
    #print('Vorig[0,0] = '+str(Vorig[0,0]))

     
    
    # Initialization for H0 as a random matrix
    Hini = np.random.rand(cW, cV)
    Wini = np.random.rand(rV, cW) #sparse.random(rV, cW, density=0.25).toarray() 
    
    
    
    
    
    
    # Wtl = tl.tensor(Worig)
    # true_res = T.tensor(np.random.rand(10, 1))
    # b = T.dot(a, true_res)
    # atb = T.dot(T.transpose(a), b)
    # ata = T.dot(T.transpose(a), a)
    # x_hals = hals_nnls(atb, ata)[0]
    # assert_array_almost_equal(true_res, x_hals, decimal=2)
    
    
    
    
    
    # noise variance
    sigma = 0#.0001
    
    if sigma == 0:
        NbSeed = 1 # if without noise nb of noise = 0
    
    

    
    Error0 = np.zeros(NbSeed)
    Error1 = np.zeros(NbSeed)
    Error2 = np.zeros(NbSeed)     
    Error3 = np.zeros(NbSeed)
    
    NbIterStop0 = np.zeros(NbSeed)
    NbIterStop1 = np.zeros(NbSeed)
    NbIterStop2 = np.zeros(NbSeed)
    NbIterStop3 = np.zeros(NbSeed)
    
     
 
    

    for  s in range(NbSeed): #[NbSeed-1]:#
          
        
        # adding noise to the observed data
        np.random.seed(s)
        N = sigma*np.random.rand(rV,cV)
        V = Vorig + N
        
        
        NbIter_inner= 20
        tol = -1
        verbose=True
        delta=0.3

        time_start0 = time.time()
        error0, W0, H0, toc0, cnt0 = NMF_Lee_Seung(V,  Wini, Hini, NbIter, NbIter_inner,tol=tol, verbose=verbose, delta=delta)
        time0 = time.time() - time_start0
        Error0[s] = error0[-1] 
        NbIterStop0[s] = len(error0)
        
      
        
        error4, W4, H4, toc4, cnt4  = NeNMF_optimMajo(V, Wini, Hini, tol=tol, itermax=NbIter, nb_inner=NbIter_inner, verbose=verbose, use_LeeS=True, delta=delta)
        time_start1 = time.time()
        error1, W1, H1, toc1, cnt1 = NMF_proposed_Frobenius(V, Wini, Hini, NbIter, NbIter_inner, tol=tol, verbose=verbose, use_LeeS=False, delta=delta)
        time1 = time.time() - time_start1
        Error1[s] = error1[-1] 
        NbIterStop1[s] = len(error1)
         
        
        
         
        time_start2 = time.time()
        error2, W2, H2, toc2, cnt2  = Grad_descent(V , Wini, Hini, NbIter, NbIter_inner, tol=tol, verbose=verbose, delta=delta)
        time2 = time.time() - time_start1
        Error2[s] = error2[-1] 
        NbIterStop2[s] = len(error2)
        
        
        time_start3 = time.time()
        error3, W3, H3, toc3, cnt3  = NeNMF(V, Wini, Hini, tol=tol, nb_inner=NbIter_inner, itermax=NbIter, verbose=verbose, delta=delta)
        time3 = time.time() - time_start3
        Error3[s] = error3[-1]
        NbIterStop3[s] = len(error3)
    
    
    fig = plt.figure(figsize=(6,3),tight_layout = {'pad': 0})
    
    plt.semilogy(error0, label = 'Lee and Seung', linewidth = 3)
    plt.semilogy(error1,'--', label = 'Pham et al', linewidth = 3)
    plt.semilogy(error4,'--', label = 'NeNMF Pham et al', linewidth = 3)
    plt.semilogy(error2,'--', label = 'Gradient descent', linewidth = 3)   
    plt.semilogy(error3,'--', label = 'NeNMF', linewidth = 3)
    plt.title('Objective function values versus iterations', fontsize=14)# for different majorizing functions')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(r'$\log\left( || V - WH || \right)$', fontsize=14)
    plt.legend(fontsize = 14)
    plt.grid(True)

    # Moving averages
    plt.figure(figsize=(6,3),tight_layout = {'pad': 0})
    k=10
    plt.plot(np.convolve(cnt0, np.ones(k)/k, mode='valid')[::3])
    plt.plot(np.convolve(cnt1, np.ones(k)/k, mode='valid')[::3])
    plt.plot(np.convolve(cnt2, np.ones(k)/k, mode='valid')[::3])
    plt.plot(np.convolve(cnt3, np.ones(k)/k, mode='valid')[::3])
    plt.plot(np.convolve(cnt4, np.ones(k)/k, mode='valid')[::3])
    plt.legend(["LeeSeung", "Proposed", "GD", "NeNMF", "Proposed NeNMF"])


    plt.show() 
    
    print('Lee and Seung: Error = '+str(np.mean(Error0)) + '; NbIter = '  + str(np.mean(NbIterStop0)) + '; Elapsed time = '+str(time0)+ '\n')
    print('Pham et al: Error = '+str(np.mean(Error1)) + '; NbIter = '  + str(np.mean(NbIterStop1)) + '; Elapsed time = '+str(time1)+ '\n')
    print('Gradient descent: Error = '+str(np.mean(Error2)) + '; NbIter = '  + str(np.mean(NbIterStop2)) + '; Elapsed time = '+str(time2)+ '\n')
    print('NeNMF: Error = '+str(np.mean(Error3)) + '; NbIter = '  + str(np.mean(NbIterStop3)) + '; Elapsed time = '+str(time3)+ '\n')

    
    
    
    
    

    
    
    

 
   

     

