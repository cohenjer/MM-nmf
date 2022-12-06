import numpy as np
from scipy.linalg import hadamard
import NLS_Frobenius as nls_f 
import NLS_KL as nls_kl
import matplotlib.pyplot as plt
import nn_fac
import pandas as pd
import utils
#from nn_fac.nmf import nmf as nmf_hals
from tensorly.tenalg.proximal import fista
import soundfile as sf
from scipy import signal
import scipy.io
import time

from shootout.methods import runners as rn
from shootout.methods import post_processors as pp


'''
    We load an hyperspectral image called Urban. It has 162 clean spectral bands, and 307x307 pixels. We also load a set of good endmembers considered as ``Ground Truth'' (rank=6 spectra), and we define subsets of the image that are likely to contain pure pixels.

    For the NNLS part, we can use either W as the ground truth (tall matrix) or candidates pixels (if more than 162, fat matrix), and estimate abundances H that should be very sparse.

    For the NMF, we estimate both W and H, and plot the results. In this experiment, the Frobenius norm is well adapted.
'''

#-------------------------------------------------------------------------
# Data import and preprocessing

# Loading the data
dict = scipy.io.loadmat('./data_and_scripts/Urban.mat')

# dict is a python dictionnary. It contains the matrix we want to NMF
M = np.transpose(dict['A']) # permutation because we like spectra in W

# It can be nice to normalize the data, then absolute error is also relative error
#M = M/np.linalg.norm(M, 'fro')

# Ground truth import
# https://gitlab.com/nnadisic/giant.jl/-/blob/master/xp/data/Urban_Ref.mat
Wref = scipy.io.loadmat('./data_and_scripts/Urban_Ref.mat')
Wref = np.transpose(Wref['References'])
Wref = Wref/np.sum(Wref, axis=0)

# ground truth rank is 6
# for good init
#Href = fista(Wref.T@M,Wref.T@Wref,tol=1e-16,n_iter_max=500)

# Candidate pixels extraction (300 randomly chosen)
# We build a larger matrix Wref2, that will be our regressor
#indices = np.random.permutation(M.shape[1])[:300]
#Wref2 = M[:, indices]
#Wref2 = Wref2/np.sum(Wref2, axis=0)

# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Solving with nonnegative least squares
#from tensorly.tenalg.proximal import fista

def one_run(NbIter = 100,
            sigma = 0.1,
            delta = 0,
            epsilon = 1e-8,
            seed = 1,
            ):
    # Seeding
    rng = np.random.RandomState(seed+20)
    # Init
    Hini = pert_sigma*rng.rand(rank, n)

    error0, H0, toc0 = nls_f.NMF_proposed_Frobenius(M, Wref2, Hini, NbIter, use_LeeS=False, delta=delta, verbose=True)
    error1, H1, toc1 = nls_f.NeNMF_optimMajo(M, Wref2, Hini, itermax=NbIter, epsilon=epsilon, verbose=True, delta=delta)
    error2, H2, toc2 = nls_f.Grad_descent(M , Wref2, Hini, NbIter,  epsilon=epsilon, verbose=True, delta=delta)
    error3, H3, toc3 = nls_f.NeNMF(M, Wref2, Hini, itermax=NbIter, epsilon=epsilon, verbose=True, delta=delta)
    H4, _, _, _, error4, toc4 = nn_fac.nnls.hals_nnls_acc(Wref2.T@Y, Wref2.T@Wref2, np.copy(Hini), maxiter=NbIter, return_error=True, delta=delta, M=M)

    return {
        "errors": [error0,error1,error2,error3,error4],
        "timings": [toc0,toc1,toc2,toc3,toc4]
    }
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# # NMF computations
#
# Explicit init
m,n = M.shape
rank = 6
beta = 2 # beta=1 data is big :(
tol = 0
NbIter = 500
NbIter_hals = 150
NbIter_inner = 10
Nb_seeds = 1
pert_sigma = 1
use_gt = 0
epsilon=1e-8

df = pd.DataFrame()

# Now compute the NMF of M
for s in range(Nb_seeds):

    # print
    print("Loop number is", s)

    # Perturbing the initialization for randomization
    Wini = use_gt*Wref + pert_sigma*np.random.rand(m, rank)
    Hini = use_gt*Href + pert_sigma*np.random.rand(rank, n)

    print('Running Lee and Seung NMF')
    error0, W0, H0, toc0 = nmf_f.NMF_Lee_Seung(M,  Wini, Hini, NbIter, NbIter_inner,tol=tol, legacy=False, epsilon=epsilon)
    print('Running our proposed NeNMF')
    error1, W1, H1, toc1  = nmf_f.NeNMF_optimMajo(M, Wini, Hini, tol=tol, itermax=NbIter, nb_inner=NbIter_inner, epsilon=epsilon)
    print('Running Proximal Gradient Descent')
    #error1, W1, H1, toc1 = nmf_f.NMF_proposed_Frobenius(V, Wini, Hini, NbIter, NbIter_inner, tol=tol)
    error2, W2, H2, toc2  = nmf_f.Grad_descent(M , Wini, Hini, NbIter, NbIter_inner, tol=tol, epsilon=epsilon)
    print('Running NeNMF')
    error3, W3, H3, toc3  = nmf_f.NeNMF(M, Wini, Hini, tol=tol, nb_inner=NbIter_inner, itermax=NbIter, epsilon=epsilon)
    # Fewer max iter cause too slow
    print('Running HALS')
    W4, H4, error4, toc4 = nn_fac.nmf.nmf(M, rank, init="custom", U_0=np.copy(Wini), V_0=np.copy(Hini), n_iter_max=NbIter_hals, tol=tol, update_rule='hals',beta=2, return_costs=True, NbIter_inner=NbIter_inner)
    
    dic= {
            "batch_size": 5, # number of algorithms in each comparison run
            "method": ["NMF_LeeSeung (modern)",  "NeNMF_optimMajorant", "PGD", "NeNMF", "HALS"],
            "m": m,
            "n": n,
            "r": rank,
            "seed_idx": s,
            "epsilon": epsilon,
            "pert_sigma": pert_sigma,
            "use_gt": use_gt,
            "NbIter": NbIter,
            "NbIter_inner": NbIter_inner,
            "NbIter_hals": NbIter_hals,
            "tol": tol,
            "final_error": [error0[-1], error1[-1], error2[-1], error3[-1], error4[-1]],
            "total_time": [toc0[-1], toc1[-1], toc2[-1], toc3[-1], toc4[-1]],
            "full_error": [error0,error1,error2,error3,error4],
            "full_time": [toc0,toc1,toc2,toc3,toc4],
            "NbIterStop": [len(error0),len(error1),len(error2),len(error3),len(error4)]
        }

    df = pd.concat([df,pd.DataFrame(dic)], ignore_index=True)



# Winner at given threshold plots
min_thresh = np.log10(error0[0])
max_thresh = np.log10(error1[-1])
thresh = np.logspace(min_thresh,max_thresh-1,50)
scores_time, scores_it, timings, iterations = utils.find_best_at_all_thresh(df,thresh, 5)

fig0 = plt.figure()
plt.subplot(121)
plt.semilogx(thresh, scores_time.T)
plt.legend(["NMF_LeeSeung",  "NeNMF_optimMajorant", "PGD", "NeNMF", "HALS"])
plt.title('How many times each algorithm reached threshold the fastest (time)')
plt.xlabel('Rec error threshold')
plt.ylabel('Number of faster runs')
plt.subplot(122)
plt.semilogx(thresh, scores_it.T)
plt.legend(["NMF_LeeSeung",  "NeNMF_optimMajorant", "PGD", "NeNMF", "HALS"])
plt.title('How many times each algorithm reached threshold the fastest (iters)')
plt.xlabel('Rec error threshold')
plt.ylabel('Number of faster runs')

# Error plots
fig_convergence_plots = plt.figure()
plt.semilogy(toc0,error0, label="Lee Seung NMF")
plt.semilogy(toc1,error1, label="Phan NeNMF")
plt.semilogy(toc2,error2, label="PGD")
plt.semilogy(toc3,error3, label="NeNMF")
plt.semilogy(toc4,error4, label="HALS")
plt.legend(fontsize = 14)

fig_convergence_plots_it = plt.figure()
plt.semilogy(error0, label="Lee Seung NMF")
plt.semilogy(error1, label="Phan NeNMF")
plt.semilogy(error2, label="PGD")
plt.semilogy(error3, label="NeNMF")
plt.semilogy(error4, label="HALS")
plt.legend(fontsize = 14)
plt.show()

#---------------------------------------------------
# # Post-processing
# print('HALS time per iteration:', tHALS/100)
# #print('MM time per iteration:', tmm/n_iter_max)
# # Reconstruct M to compute relative error
# W = out[0]
# H = out[1]
# Mest = W@H
# err = np.linalg.norm(M - Mest, 'fro')
# print(err)
# # Reconstruction with mm
# #Wmm = out2[1]
# #Hmm = out2[2]
# #Mestmm = Wmm@Hmm
# #
# #errmm = np.linalg.norm(M - Mestmm, 2)
# #print(errmm)
# # Printing the convergence plot
# plt.figure()
# # no initial error in HALS error
# xhals = np.linspace(0,tHALS,n_iter_max)
# #xmm = np.linspace(0,tmm,n_iter_max)
# plt.semilogy(xhals,out[2])
# #plt.semilogy(xmm,out2[0][1:])
# plt.xlabel('fake time (iter/avg iter time)')
# plt.legend('HALS')
# plt.show()
# # Normalisations
# W = W/np.sqrt(np.sum(W**2,axis=0))
# #Wmm = Wmm/np.sqrt(np.sum(Wmm**2,axis=0))
# # Visualization of the results
# # TODO: PERMUTE COLUMNS SO THAT THEY MATCH
# plt.subplot(3,6,3)
# plt.plot(W)
# plt.legend([1,2,3,4,5,6])
# #plt.subplot(3,5,3)
# #plt.plot(Wmm)
# #plt.legend([1,2,3,4,5])
# plt.subplot(367)
# plt.imshow(np.transpose(np.reshape(H[0, :], [307,307])))
# plt.title('1')
# plt.subplot(368)
# plt.imshow(np.transpose(np.reshape(H[1, :], [307,307])))
# plt.title('2')
# plt.subplot(369)
# plt.imshow(np.transpose(np.reshape(H[2, :], [307,307])))
# plt.title('3')
# plt.subplot(3,6,10)
# plt.imshow(np.transpose(np.reshape(H[3, :], [307,307])))
# plt.title('4')
# plt.subplot(3,6,11)
# plt.imshow(np.transpose(np.reshape(H[4, :], [307,307])))
# plt.title('5')
# plt.subplot(3,6,12)
# plt.imshow(np.transpose(np.reshape(H[5, :], [307,307])))
# plt.title('6')
# #plt.subplot(3,5,11)
# #plt.imshow(np.transpose(np.reshape(Hmm[0, :], [307,307])))
# #plt.title('1')
# #plt.subplot(3,5,12)
# #plt.imshow(np.transpose(np.reshape(Hmm[1, :], [307,307])))
# #plt.title('2')
# #plt.subplot(3,5,13)
# #plt.imshow(np.transpose(np.reshape(Hmm[2, :], [307,307])))
# #plt.title('3')
# #plt.subplot(3,5,14)
# #plt.imshow(np.transpose(np.reshape(Hmm[3, :], [307,307])))
# #plt.title('4')
# #plt.subplot(3,5,15)
# #plt.imshow(np.transpose(np.reshape(Hmm[4, :], [307,307])))
# #plt.title('5')
# plt.show()

## Visualization of the results
#plt.subplot(3,6,3)
#plt.plot(Wref)
#plt.legend([1,2,3,4,5,6])
##plt.subplot(3,5,3)
##plt.plot(Wmm)
##plt.legend([1,2,3,4,5])
#plt.subplot(367)
#plt.imshow(np.transpose(np.reshape(H[0, :], [307,307])))
#plt.title('1')
#plt.subplot(368)
#plt.imshow(np.transpose(np.reshape(H[1, :], [307,307])))
#plt.title('2')
#plt.subplot(369)
#plt.imshow(np.transpose(np.reshape(H[2, :], [307,307])))
#plt.title('3')
#plt.subplot(3,6,10)
#plt.imshow(np.transpose(np.reshape(H[3, :], [307,307])))
#plt.title('4')
#plt.subplot(3,6,11)
#plt.imshow(np.transpose(np.reshape(H[4, :], [307,307])))
#plt.title('5')
#plt.subplot(3,6,12)
#plt.imshow(np.transpose(np.reshape(H[5, :], [307,307])))
#plt.title('6')
#plt.show()

## With overcomplete W from randomly picked pixels
#UtM = Wref2.T@M
#UtU = Wref2.T@Wref2
#H = nn_fac.nnls.hals_nnls_acc(UtM,UtU, in_V = np.random.rand(Wref2.shape[1],M.shape[1]),delta=1e-8, nonzero=False, maxiter=100)[0]
##H = fista(UtM,UtU, tol=1e-8, n_iter_max=100)
## Postprocessing
#Mest = Wref2@H
## Error computation
#print('Relative Frobenius error with randomly picked pixels, FISTA alg', np.linalg.norm(M - Mest2, 'fro'))