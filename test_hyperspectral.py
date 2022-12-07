import numpy as np
from scipy.linalg import hadamard
import NMF_Frobenius as nmf_f 
import NMF_KL as nmf_kl
import matplotlib.pyplot as plt
import plotly.express as px
import nn_fac
import pandas as pd
import utils
#from nn_fac.nmf import nmf as nmf_hals
from tensorly.tenalg.proximal import fista
import soundfile as sf
from scipy import signal
import scipy.io
import time
from shootout.methods.runners import run_and_track
from shootout.methods.post_processors import find_best_at_all_thresh, df_to_convergence_df, error_at_time_or_it
from shootout.methods.plotters import plot_speed_comparison

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
m,n = M.shape

# It can be nice to normalize the data, then absolute error is also relative error
#M = M/np.linalg.norm(M, 'fro')

# Ground truth import
# https://gitlab.com/nnadisic/giant.jl/-/blob/master/xp/data/Urban_Ref.mat
Wref = scipy.io.loadmat('./data_and_scripts/Urban_Ref.mat')
Wref = np.transpose(Wref['References'])

# ground truth rank is 6
# for good init
#Href = fista(Wref.T@M,Wref.T@Wref,tol=1e-16,n_iter_max=500)


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Solving with nonnegative least squares
#from tensorly.tenalg.proximal import fista

algs = ["Proposed_l2_delta1.8", "Proposed_l2_extrapolated", "GD_l2", "NeNMF_l2", "HALS", "Lee_Sung_KL", "Proposed_KL"]
name = "hsi_nls_test_06_12_2022"
Nb_seeds = 5
@run_and_track(
    nb_seeds=Nb_seeds,
    algorithm_names=algs, 
    path_store="Results/",
    name_store=name,
)
def one_run(rank = 6,
            NbIter = 100,
            NbIter_inner = 100,
            delta=0.1,
            epsilon = 1e-8,
            tol=0,
            seed = 1,
            ):
    # Seeding
    rng = np.random.RandomState(seed+20)
    # Init
    Wini = Wref + 0.1*np.random.rand(m,rank)
    Hini = rng.rand(rank, n)

    # Frobenius algorithms
    #error0, W0, H0, toc0, cnt0 = nmf_f.NMF_Lee_Seung(M,  Wini, Hini, NbIter, NbIter_inner,tol=tol, legacy=False, epsilon=epsilon, verbose=True, delta=delta)   
    error0, W0, H0, toc0, cnt0 = nmf_f.NMF_proposed_Frobenius(M, Wini, Hini, NbIter, NbIter_inner, tol=tol, use_LeeS=False, delta=delta, verbose=True, print_it=1)
    error1, W1, H1, toc1, cnt1  = nmf_f.NeNMF_optimMajo(M, Wini, Hini, tol=tol, itermax=NbIter, nb_inner=NbIter_inner, epsilon=epsilon, verbose=True, delta=delta, print_it=1)
    error2, W2, H2, toc2, cnt2  = nmf_f.Grad_descent(M , Wini, Hini, NbIter, NbIter_inner, tol=tol, epsilon=epsilon, verbose=True, delta=delta, print_it=1)
    error3, W3, H3, toc3, cnt3  = nmf_f.NeNMF(M, Wini, Hini, tol=tol, nb_inner=NbIter_inner, itermax=NbIter, epsilon=epsilon, verbose=True, delta=delta, print_it=1)
    W4, H4, error4, toc4, cnt4 = nn_fac.nmf.nmf(M, rank, init="custom", U_0=np.copy(Wini), V_0=np.copy(Hini), n_iter_max=NbIter, tol=tol, update_rule='hals',beta=2, return_costs=True, NbIter_inner=NbIter_inner, verbose=True, delta=delta)

    # KL algorithms
    error5, W5, H5, toc5, cnt5 = nmf_kl.Lee_Seung_KL(M, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=True, print_it=1)
    error6, W6, H6, toc6, cnt6 = nmf_kl.Proposed_KL(M, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=True, print_it=1)

    return {
        "errors": [error0,error1,error2,error3,error4, error5, error6],
        "timings": [toc0,toc1,toc2,toc3,toc4,toc5,toc6],
        "loss": 5*["l2"]+2*["kl"],
    }

df = pd.read_pickle("Results/"+name)

# Making a convergence plot dataframe
df_l2_conv = df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               filters={"loss":"l2"})

df_kl_conv = df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               filters={"loss":"kl"})
# ----------------------- Plot --------------------------- #
# Convergence plots with all runs
pxfig = px.line(df_l2_conv, line_group="groups", x="timings", y= "errors", color='algorithm', 
            log_y=True,
            height=1000)
pxfig.update_layout(
    font_size = 20,
    width=1200, # in px
    height=900,
    )
pxfig2 = px.line(df_kl_conv, line_group="groups", x="timings", y= "errors", color='algorithm',
            log_y=True,
            height=1000)
pxfig2.update_layout(
    font_size = 20,
    width=1200, # in px
    height=900,
    )
pxfig.show()
pxfig2.show()

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