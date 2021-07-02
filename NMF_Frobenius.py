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

def NMF_Lee_Seung(Vorig, V, W0, H0, NbIter):

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
        H = (H*(W.T.dot(V)))/ (W.T.dot(WH))
        WH = W.dot(H)

        # FIXED H ESTIMATE W
        W = (W*(V.dot(H.T)))/ (WH.dot(H.T))

        WH = W.dot(H)
        # compute the error
        error.append(la.norm(Vorig- WH)/error_norm)
        # check if the err is small enough to stop
        if (error[k] < 1e-7):

            return error, W, H

    return error, W, H

#------------------------------------
#  NMF algorithm proposed version

def NMF_proposed_Frobenius(Vorig, V , W0, H0, NbIter, NbIter_inner):

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
    gamma = 1.9  # arbitrary??
    #error_norm = np.prod(Vorig.shape)
    error = [la.norm(Vorig-W.dot(H))**2]  #/error_norm]
    for k in range(NbIter):

        # FIXED W ESTIMATE H
        A1 = W.T.dot(W)
        Hess1 = lambda X: A1.dot(X)
        B1 = W.T.dot(V)
        # find the zero entries of B
        # Independent of X, computation time could be saved
        ind = np.where(B1==0)
        sqrtB1 = np.sqrt(B1)

        for ih in range(NbIter_inner):
            # no need for side function
            #aux_H = auxiliary(Hess1, B1, H, ind, sqrtB)
            aux_H = sqrtB1/Hess1(sqrtB1)
            aux_H[ind] = 0
            #H =  np.maximum(H + gamma*aux_H*(B1 - Hess1(H)),0)
            H =  np.maximum(H + gamma*aux_H*(B1 - Hess1(H)),0)
            # ii = np.all((H>=0))
            # if ~ii:
            #     print('algo stop at '+str(k))
            #     return err, H, W

        # FIXED H ESTIMATE W
        A2 = H.dot(H.T)
        Hess2 = lambda X: X.dot(A2)
        B2 = V.dot(H.T)
        # Independent of X, computation time could be saved
        ind = np.where(B2==0)
        sqrtB2 = np.sqrt(B2)
        for iw in range(NbIter_inner):
            aux_W = sqrtB2/Hess2(sqrtB2)
            aux_W[ind] = 0
            #aux_W =  auxiliary(Hess2, B2, W)
            W = np.maximum(W + gamma*aux_W*(B2 - Hess2(W)),0)

        # error calc can be improved using A2 and B2
        error.append(la.norm(Vorig- W.dot(H)) **2)  #/error_norm)
        print('reconstruction error', error[-1])
        # Check if the error is smalle enough to stop the algorithm
        #if (error[k] <1e-7):
        #    print('algo stop at iteration =  '+str(k))
        #    return error, W, H


    return error, W, H

###------ define the SPD matrix that satisfies the auxiliary function

def auxiliary(Hess, B, X, indB, sqrtB):
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
    # Independent of X, computation time could be saved
    ind = np.where(B==0)
    sqrtB = np.sqrt(B)
    A = sqrtB/Hess(sqrtB)
    A[ind] = 0
    return A






################## Gradient descent method

def Grad_descent(Vorig, V , W0, H0, NbIter, NbIter_inner):

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

        # Check if the error is smalle enough to stop the algorithm
        if (error[k] <1e-7):

            return error, W, H#, (k+inner_iter_total)


    return error, W, H #, (k+inner_iter_total)



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


    NbIterStop0 = np.zeros(NbSeed)
    NbIterStop1 = np.zeros(NbSeed)
    NbIterStop2 = np.zeros(NbSeed)





    for  s in range(NbSeed): #[NbSeed-1]:#


        # adding noise to the observed data
        np.random.seed(s)
        N = sigma*np.random.rand(rV,cV)
        V = Vorig + N



        time_start0 = time.time()
        error0, W0, H0 = NMF_Lee_Seung(Vorig, V,  Wini, Hini, NbIter)
        time0 = time.time() - time_start0
        Error0[s] = error0[-1]
        NbIterStop0[s] = len(error0)


        NbIter_inner= 10

        time_start1 = time.time()
        error1, W1, H1  = NMF_proposed_Frobenius(Vorig, V, Wini, Hini, NbIter, NbIter_inner)
        time1 = time.time() - time_start1
        Error1[s] = error1[-1]
        NbIterStop1[s] = len(error1)




        time_start2 = time.time()
        error2, W2, H2  = Grad_descent(Vorig, V , Wini, Hini, NbIter, NbIter_inner)
        time2 = time.time() - time_start1
        Error2[s] = error2[-1]
        NbIterStop2[s] = len(error2)



    fig = plt.figure(figsize=(6,3),tight_layout = {'pad': 0})
    plt.semilogy(error0, label = 'Lee and Seung', linewidth = 3)
    plt.semilogy(error1,'--', label = 'Pham et al', linewidth = 3)
    plt.semilogy(error2,'--', label = 'Gradient descent', linewidth = 3)

    print('Lee and Seung: Error = '+str(np.mean(Error0)) + '; NbIter = '  + str(np.mean(NbIterStop0)) + '; Elapsed time = '+str(time0)+ '\n')
    print('Pham et al: Error = '+str(np.mean(Error1)) + '; NbIter = '  + str(np.mean(NbIterStop1)) + '; Elapsed time = '+str(time1)+ '\n')
    print('Gradient descent: Error = '+str(np.mean(Error2)) + '; NbIter = '  + str(np.mean(NbIterStop2)) + '; Elapsed time = '+str(time2)+ '\n')
