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


#------------------------------------
# PMF algorithm version Lee and Seung

def NMF_Lee_Seung(Vorig, V, W0, H0, NbIter, NbIter_inner, legacy=True, epsilon=1e-8, tol=1e-7):
    
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
    legacy: bool
        If True, implements the original update rule of Lee and Seung.
        If False, uses max( update, epsilon ) which ensures convergence with the BSUM framework and avoids zero-locking.
    epsilon: float
        if legacy is False, factors satisfy H > epsilon, W > epsilon instead of elementwise nonnegativity.
    tol: float
        stopping criterion, algorithm stops if error<tol.

    Returns
    -------
    error : darray
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
    error = [la.norm(Vorig- WH)/error_norm]
     
    
    for k in range(NbIter):
        
        # FIXED W ESTIMATE H      
        WtW = W.T@W
        WtV = W.T@V
        for j in range(NbIter_inner): 
            if legacy:
                H = (H*WtV)/ (WtW@H)
            else:
                H = np.maximum((H*WtV)/ (WtW@H), epsilon)
        
        # FIXED H ESTIMATE W
        VHt = V@H.T
        HHt = H@H.T
        for j in range(NbIter_inner):
            if legacy:
                W = (W*VHt)/ (W@HHt)
            else: 
                W = np.maximum((W*VHt)/ (W@HHt), epsilon)

        # compute the error
        error.append(la.norm(Vorig- W@H)/error_norm)
        # check if the err is small enough to stop 
        if (error[-1] < tol):
            #if not legacy:
            #   # Putting zeroes where we thresholded with epsilon
            #   W[W==epsilon]=0 
            #   H[H==epsilon]=0
            return error, W, H

    return error, W, H

#------------------------------------
#  NMF algorithm proposed version
 
def NMF_proposed_Frobenius(Vorig, V , W0, H0, NbIter, NbIter_inner, tol=1e-7, epsilon=1e-8):
    
    """
    The goal of this method is to factorize (approximately) the non-negative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in miminime (1/2) || V - WH ||^2 s.t. W, H >= 0
    
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
    tol: float
        stopping criterion, algorithm stops if error<tol.
    
    Returns
    -------
    err : darray
        vector that saves the error between Vorig with WH at each iteration.
    H : RxN array
        non-negative estimated matrix.
    W : MxR array
        non-negative estimated matrix.

    """
    
    W = W0.copy()
    H = H0.copy()
    gamma = 1.9
    error_norm = np.prod(Vorig.shape)
    error = [la.norm(Vorig-W.dot(H))/error_norm]
    for k in range(NbIter):
        
        # FIXED W ESTIMATE H
        A1 = W.T.dot(W)
        Hess1 = lambda X: A1.dot(X)
        B1 = W.T.dot(V)
        for ih in range(NbIter_inner):
            aux_H = auxiliary(Hess1, B1, H)   
            H =  np.maximum(H + gamma*aux_H*(B1 - Hess1(H)), epsilon)
            # ii = np.all((H>=0))
            # if ~ii:
            #     print('algo stop at '+str(k))
            #     return err, H, W
            
        # FIXED H ESTIMATE W
        A2 = H.dot(H.T)
        Hess2 = lambda X: X.dot(A2)
        B2 = V.dot(H.T)
        for iw in range(NbIter_inner):
            aux_W =  auxiliary(Hess2, B2, W) 
            W = np.maximum(W + gamma*aux_W*(B2 - Hess2(W)), epsilon)
               
        error.append(la.norm(Vorig- W.dot(H))/error_norm)
        # Check if the error is smalle enough to stop the algorithm 
        if (error[-1] <tol):            
            print('algo stop at iteration =  '+str(k))
            return error, W, H
            
        
    return error, W, H 

###------ define the SPD matrix that satisfies the auxiliary function

def auxiliary(Hess, B, X):
    """
    Define the SPD matrix A that satisfies the majoration condition for psi that has Hess operator of hessian 
    Denote B = Hess(X) when B_ij = 0  we can choose A_ij = 0 and X_ij = 0
    We can suppose that B_ij ~= 0 for all ij
    To define the auxiliary matrix we need to 
    find X* that minimizes || B/X ||_1 such that X>=0 and || X|| = 1
    The solution of this optimization problem is X* = sqrt(B)/ ||sqrt(B)|| 
    Therefore the majorant matrix A = Hess(sqrt(B))/||sqrt(B)||

    Parameters
    ----------
    Hess : operator from R^N --> R^{NxN} 
        to define the hessian matrix of the cost function 
        
    B : NxM array
        non-negative matrix.
   X : NxM array 
       non-negative matrx.
     
    Returns
    -------
    A : NxM array  
        non-negative matrix that minimize || B/X ||_1 s.t. || X|| = 1
    """
    
     # find the zero entries of B
    ind = np.where(B==0)
    sqrtB = np.sqrt(B)
    
    A = sqrtB/Hess(sqrtB)
    A[ind] = 0 
    return A
    
    
    
 


################## Gradient descent method

def Grad_descent(Vorig, V , W0, H0, NbIter, NbIter_inner, tol=1e-7):
    
    """"
    The goal of this method is to factorize (approximately) the non-negative (entry-wise) matrix V by WH i.e
    V = WH + N where N represents to the noise --> It leads to find W,H in miminime (1/2) || V - WH ||^2 s.t. W, H >= 0
    
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

    """
    
    
    
    W = W0.copy()
    H = H0.copy()
    error_norm = np.prod(Vorig.shape)
    error = [la.norm(Vorig- W.dot(H))/error_norm]
     
    inner_iter_total = 0 
    
    for k in range(NbIter):
        # FIXED W ESTIMATE H
        Aw = W.T.dot(W)      
        normAw = la.norm(Aw,2)
        WtV = W.T.dot(V)
        for ih in range(NbIter_inner):          
            H =  np.maximum(H + (1.9/normAw)*(WtV - Aw.dot(H)),1e-7)
        inner_iter_total +=NbIter_inner
          
        # FIXED H ESTIMATE W
        Ah = H.dot(H.T)
        normAh = la.norm(Ah,2)
        VHt = V.dot(H.T)
        for iw in range(NbIter_inner):       
            W = np.maximum(W + (1.9/normAh)*(VHt - W.dot(Ah)),1e-7)
        inner_iter_total +=NbIter_inner
        
        # compute the error 
        error.append(la.norm(Vorig- W.dot(H))/error_norm)
        
        # Check if the error is small enough to stop the algorithm 
        if (error[-1] <tol):
        
            return error, W, H#, (k+inner_iter_total)
            
        
    return error, W, H #, (k+inner_iter_total)




#####-------------------------------------------------------------
# NeNMF
# from https://www.academia.edu/7815546/NeNMF_An_Optimal_Gradient_Method_for_Nonnegative_Matrix_Factorization
#-------------------------------------------------------------
# created: # 2021 oct. 11
#-------------------------------------------------------------

 

def OGM_H(V,W,H, Aw, L, nb_inner):
        # V≈WH, W≥O, H≥0
        # updates H        
        
        Y = H.copy()
        alpha     = 1
        #Aw = W.T.dot(W)
        #L = la.norm(Aw,2)
        WtV = W.T.dot(V)
         
        for ih in range(nb_inner):
            H_ = H.copy()
            alpha_ = alpha 
            H = np.maximum(Y+L*(WtV - Aw.dot(Y)),0) # projection entrywise on R+ of gradient step
            alpha = (1+np.sqrt(4*alpha**2+1))/2  # Nesterov momentum parameter           
            Y = H + ((alpha-1)/alpha_)*(H-H_)
            
        
        return H

def OGM_W(V,W,H, Ah, L, nb_inner):
        # V≈WH, W≥O, H≥0
        # updates W
        # eps: threshold for stopping criterion
        
        #Ah = H.dot(H.T)
        #L = la.norm(Ah,2)
        VHt = V.dot(H.T)
        alpha = 1
        Y = W.copy()
        for iw in range(nb_inner):
            W_ = W.copy()
            alpha_ = alpha 
            W = np.maximum(Y + L*(VHt - Y.dot(Ah)),0)
            alpha = (1+np.sqrt(4*alpha**2+1))/2  # Nesterov momentum parameter           
            Y = W + ((alpha-1)/alpha_)*(W-W_)
        return W
            
        
         
def NeNMF(Vorig, V, W0, H0, tol=1e-7, nb_inner=10, itermax=10000):
    W = W0.copy()
    H = H0.copy()
    error_norm = np.prod(Vorig.shape)
    error = [la.norm(Vorig- W.dot(H))/error_norm]
    #inner_iter_total = 0
    #test   = 1 # 
    #while (test> tol): 
    iter = 1
    while (error[-1]> tol) and (iter<itermax): 

        Aw = W.T.dot(W)
        Lw = 1/la.norm(Aw,2)
        H     = OGM_H(V, W, H, Aw, Lw, nb_inner)
        
        Ah = H.dot(H.T)
        Lh = 1/la.norm(Ah,2)
        W     = OGM_W(V, W, H, Ah, Lh, nb_inner)
        #inner_iter_total = inner_iter_total+20
        #WH = W.dot(H)
        #dH,dW = -W.T.dot(V-WH) , (V - WH).dot(H.T)
        #test  = la.norm(dH*(H>0) + np.minimum(dH,0)*(H==0), 2) +la.norm(dW*(W>0) + np.minimum(dW,0)*(W==0), 2) # eq. 21 p.2885 -> A RETRAVAILLER
        #error.append(la.norm(Vorig- WH)/error_norm)
        error.append(la.norm(Vorig- W@H)/error_norm)
        iter+=1
        
    
    return error, W, H#, inner_iter_total


def NeNMF_optimMajo(Vorig, V, W0, H0, tol=1e-7, nb_inner=10, itermax = 10000):
    W = W0.copy()
    H = H0.copy()
    error_norm = np.prod(Vorig.shape)
    error = [la.norm(Vorig- W.dot(H))/error_norm]
    #inner_iter_total = 0
    #test   = 1 # 
    #while (test> tol): 
    iter = 0
    while error[-1]>tol and iter < itermax:
        
        #----fixed w estimate H
        
        A1 = W.T.dot(W)
        sqrtB1 =np.sqrt(W.T.dot(V))
        # find the zero entries of B
        # Independent of X, computation time could be saved
    
        Lw = sqrtB1/(A1.dot(sqrtB1)+1e-10)
        
        #Lw = 1/la.norm(Aw,2)
        H     = OGM_H(V, W, H, A1, Lw, nb_inner)
        
        # fixed h estimate w
        
        A2 = H.dot(H.T)
        sqrtB2 = np.sqrt(V.dot(H.T))
        # Independent of X, computation time could be saved
              
        Lh = sqrtB2/(sqrtB2.dot(A2)+1e-10)        
        W = OGM_W(V, W, H, A2, Lh, nb_inner)
        #inner_iter_total = inner_iter_total+20
        #WH = W.dot(H)
        #dH,dW = -W.T.dot(V-WH) , (V - WH).dot(H.T)
        #test  = la.norm(dH*(H>0) + np.minimum(dH,0)*(H==0), 2) +la.norm(dW*(W>0) + np.minimum(dW,0)*(W==0), 2) # eq. 
        # 21 p.2885 -> A RETRAVAILLER
        #error.append(la.norm(Vorig- WH)/error_norm)
        error.append(la.norm(Vorig- W@H)/error_norm)
        iter += 1
    
    return error, W, H#, inner_iter_total




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
        
        
        NbIter_inner= 50
        tol = 1e-8
        
        time_start0 = time.time()
        error0, W0, H0 = NMF_Lee_Seung(Vorig, V,  Wini, Hini, NbIter, NbIter_inner,tol=tol)
        time0 = time.time() - time_start0
        Error0[s] = error0[-1] 
        NbIterStop0[s] = len(error0)
        
      
        
        time_start1 = time.time()
        error1, W1, H1  = NeNMF_optimMajo(Vorig, V, Wini, Hini, tol=tol, itermax=NbIter, nb_inner=NbIter_inner)
        #error1, W1, H1 = NMF_proposed_Frobenius(Vorig, V, Wini, Hini, NbIter, NbIter_inner, tol=tol)
        time1 = time.time() - time_start1
        Error1[s] = error1[-1] 
        NbIterStop1[s] = len(error1)
         
        
        
         
        time_start2 = time.time()
        error2, W2, H2  = Grad_descent(Vorig, V , Wini, Hini, NbIter, NbIter_inner, tol=tol)
        time2 = time.time() - time_start1
        Error2[s] = error2[-1] 
        NbIterStop2[s] = len(error2)
        
        
        time_start3 = time.time()
        error3, W3, H3  = NeNMF(Vorig, V, Wini, Hini, tol=tol, nb_inner=NbIter_inner, itermax=NbIter)
        time3 = time.time() - time_start3
        Error3[s] = error3[-1]
        NbIterStop3[s] = len(error3)
    
    
    fig = plt.figure(figsize=(6,3),tight_layout = {'pad': 0})
    
    plt.semilogy(error0, label = 'Lee and Seung', linewidth = 3)
    plt.semilogy(error1,'--', label = 'Pham et al', linewidth = 3)
    plt.semilogy(error2,'--', label = 'Gradient descent', linewidth = 3)   
    plt.semilogy(error3,'--', label = 'NeNMF', linewidth = 3)
    plt.title('Objective function values versus iterations', fontsize=14)# for different majorizing functions')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(r'$\log\left( || V - WH || \right)$', fontsize=14)
    plt.legend(fontsize = 14)
    plt.grid(True)
    
    
    
    print('Lee and Seung: Error = '+str(np.mean(Error0)) + '; NbIter = '  + str(np.mean(NbIterStop0)) + '; Elapsed time = '+str(time0)+ '\n')
    print('Pham et al: Error = '+str(np.mean(Error1)) + '; NbIter = '  + str(np.mean(NbIterStop1)) + '; Elapsed time = '+str(time1)+ '\n')
    print('Gradient descent: Error = '+str(np.mean(Error2)) + '; NbIter = '  + str(np.mean(NbIterStop2)) + '; Elapsed time = '+str(time2)+ '\n')
    print('NeNMF: Error = '+str(np.mean(Error3)) + '; NbIter = '  + str(np.mean(NbIterStop3)) + '; Elapsed time = '+str(time3)+ '\n')

    
    
    
    
    

    
    
    

 
   

     

