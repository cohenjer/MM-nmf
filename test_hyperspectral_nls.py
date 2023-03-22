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
import plotly.express as px
import time
import sys
from shootout.methods import runners as rn
from shootout.methods import post_processors as pp
import plotly.io as pio
pio.kaleido.scope.mathjax = None # bugggggg
pio.templates.default= "plotly_white"



'''
    We load an hyperspectral image called Urban. It has 162 clean spectral bands, and 307x307 pixels. We also load a set of good endmembers considered as ``Ground Truth'' (rank=6 spectra), and we define subsets of the image that are likely to contain pure pixels.

    For the NNLS part, we can use either W as the ground truth (tall matrix) or candidates pixels (if more than 162, fat matrix), and estimate abundances H that should be very sparse.

    For the NMF, we estimate both W and H, and plot the results. In this experiment, the Frobenius norm is well adapted.
'''

#-------------------------------------------------------------------------
# Data import and preprocessing

# Loading the data
dico = scipy.io.loadmat('./data_and_scripts/Urban.mat')

# dict is a python dictionnary. It contains the matrix we want to NMF
M = np.transpose(dico['A']) # permutation because we like spectra in W

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

algs = ["fastMU_Fro", "fastMU_Fro_ex", "GD_Fro", "NeNMF_Fro", "MU_Fro", "HALS", "MU_KL", "fastMU_KL", "fastMU_KL_approx"]
name = "hsi_nls_test_02_14_2023"
if len(sys.argv)==1:
    nb_seeds = 0 #no run
else:
    nb_seeds = int(sys.argv[1])  # Change this to >0 to run experiments

@rn.run_and_track(
    nb_seeds=nb_seeds,
    algorithm_names=algs, 
    path_store="Results/",
    name_store=name,
    seeded_fun=True)
def one_run(NbIter = 100,
            delta = 0,
            verbose=False,
            epsilon = 1e-16,
            seed = 1,
            ):
    # Seeding
    rng = np.random.RandomState(seed+20)
    # Init
    Hini = rng.rand(Wref.shape[1], M.shape[1])

    error0, H0, toc0 = nls_f.NMF_proposed_Frobenius(M, Wref, np.copy(Hini), NbIter, use_LeeS=False, delta=delta, verbose=verbose, gamma=1.9, epsilon=epsilon)
    #error1, H1, toc1 = nls_f.NMF_proposed_Frobenius(M, Wref, Hini, NbIter, use_LeeS=True, delta=delta, verbose=verbose, gamma=1, epsilon=epsilon)
    error2, H2, toc2 = nls_f.NeNMF_optimMajo(M, Wref, Hini, itermax=NbIter, epsilon=epsilon, verbose=verbose, delta=delta, gamma=1)
    error3, H3, toc3 = nls_f.Grad_descent(M , Wref, Hini, NbIter,  epsilon=epsilon, verbose=verbose, delta=delta, gamma=1.9)
    error4, H4, toc4 = nls_f.NeNMF(M, Wref, Hini, itermax=NbIter, epsilon=epsilon, verbose=verbose, delta=delta)
    error5, H5, toc5 = nls_f.NMF_Lee_Seung(M,  Wref, Hini, NbIter, legacy=False, delta=delta, verbose=verbose, epsilon=epsilon)
    
    # HALS is unfair because we compute things before. We add the time needed for this back after the algorithm
    tic = time.perf_counter()
    WtV = Wref.T@M
    WtW = Wref.T@Wref
    toc6_offset = time.perf_counter() - tic
    H6, _, _, _, error6, toc6 = nn_fac.nnls.hals_nnls_acc(WtV, WtW, np.copy(Hini), maxiter=NbIter, return_error=True, delta=delta, M=M)
    toc6 = [toc6[i] + toc6_offset for i in range(len(toc6))] # leave the 0 in place for init
    toc6[0]=0
    
    error7, H7, toc7 = nls_kl.Lee_Seung_KL(M, Wref, Hini, NbIter=NbIter, verbose=verbose, delta=delta, epsilon=epsilon)
    error8, H8, toc8 = nls_kl.Proposed_KL(M, Wref, Hini, NbIter=NbIter, verbose=verbose, delta=delta, use_LeeS=False, gamma=1.9, epsilon=epsilon, true_hessian=True)
    error9, H9, toc9 = nls_kl.Proposed_KL(M, Wref, Hini, NbIter=NbIter, verbose=verbose, delta=delta, use_LeeS=False, gamma=1.9, epsilon=epsilon, true_hessian=False)


    return {
        "errors": [error0,error2,error3,error4, error5, error6, error7, error8, error9],
        "timings": [toc0,toc2,toc3,toc4, toc5, toc6, toc7, toc8, toc9],
        "loss": 6*["l2"]+3*["kl"],
    }


df = pd.read_pickle("Results/"+name)

# Interpolating
df = pp.interpolate_time_and_error(df, npoints = 100, adaptive_grid=True)

# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
df_l2_conv = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               filters={"loss":"l2"}, err_name="errors_interp", time_name="timings_interp")
df_l2_conv = df_l2_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})

df_l2_conv_it = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               filters={"loss":"l2"})

df_kl_conv = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               filters={"loss":"kl"}, err_name="errors_interp", time_name="timings_interp")
df_kl_conv = df_kl_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
# ----------------------- Plot --------------------------- #
#fig_winner = plot_speed_comparison(thresh, scores_time, scores_it, legend=algs)
#fig_winner.show()

# Median plots
df_l2_conv_median_time = pp.median_convergence_plot(df_l2_conv, type="timings")
df_l2_conv_median_it = pp.median_convergence_plot(df_l2_conv_it, type="iterations")
df_kl_conv_median_time = pp.median_convergence_plot(df_kl_conv, type="timings")

# Convergence plots with all runs
pxfig = px.line(df_l2_conv_median_time, 
            x="timings", 
            y= "errors", 
            color='algorithm',
            line_dash='algorithm',
            log_y=True,
            #error_y="q_errors_p", 
            #error_y_minus="q_errors_m", 
)

# Final touch
pxfig.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfig.update_layout(
    title_text = "NLS",
    font_size = 12,
    width=450*1.62/2, # in px
    height=450,
    xaxis=dict(range=[0,1.0], title_text="Time (s)"),
    yaxis=dict(title_text="Fit")
)

pxfig.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfig.update_yaxes(
    matches=None,
    showticklabels=True
)
# buuuuuugggedddd
pxfig.write_image("Results/"+name+"_fro.pdf")
pxfig.show()

pxfig2 = px.line(df_kl_conv_median_time, 
            x="timings", 
            y= "errors", 
            color='algorithm',
            line_dash='algorithm',
            log_y=True,
            #error_y="q_errors_p", 
            #error_y_minus="q_errors_m", 
)

# Final touch
pxfig2.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfig2.update_layout(
    title_text = "NLS",
    font_size = 12,
    width=450*1.62/2, # in px
    height=450,
    xaxis=dict(title_text="Time (s)"),
    yaxis=dict(title_text="Fit")
)

pxfig2.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfig2.update_yaxes(
    matches=None,
    showticklabels=True
)

pxfig2.write_image("Results/"+name+"_kl.pdf")
pxfig2.show()


## Interpolating
#df = pp.interpolate_time_and_error(df, npoints = 40, adaptive_grid=True)

## Making a convergence plot dataframe
## We will show convergence plots for various sigma values, with only n=100
#df_conv_time = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               #err_name="errors_interp", time_name="timings_interp")
#df_conv_time = df_conv_time.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
## iteration plots
#df_conv_it = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=[])

## ----------------------- Plot --------------------------- #
## Median plots
#df_conv_median_time = pp.median_convergence_plot(df_conv_time, type="timings")
#df_conv_median_it = pp.median_convergence_plot(df_conv_it, type="iterations")
## subsample
##df_conv_median_it = df_conv_median_it[df_conv_median_it["it"]%3==0] # manual tweak

## Convergence plots with all runs
#pxfig = px.line(df_conv_median_time, 
            #x="timings", 
            #y= "errors", 
            #color='algorithm',
            #log_y=True,
            #error_y="q_errors_p", 
            #error_y_minus="q_errors_m", 
            #template="plotly_white",
            #height=1000)
#pxfig.update_layout(
    #font_size = 20,
    #width=1200, # in px
    #height=900,
    #)
#pxfig.update_traces(
    #selector=dict(),
    #line_width=3,
    #error_y_thickness = 0.3)
#pxfig2 = px.line(df_conv_median_it, 
            #x="it", 
            #y= "errors", 
            #color='algorithm',
            #log_y=True,
            #error_y="q_errors_p", 
            #error_y_minus="q_errors_m", 
            #template="plotly_white",
            #height=1000)
#pxfig2.update_layout(
    #font_size = 20,
    #width=1200, # in px
    #height=900,
    #)
#pxfig2.update_traces(
    #selector=dict(),
    #line_width=3,
    #error_y_thickness = 0.3)


#pxfig2.show()
#pxfig.show()

## --------------------------------------------------------------------
## --------------------------------------------------------------------
## # NMF computations
##
## Explicit init
#m,n = M.shape
#rank = 6
#beta = 2 # beta=1 data is big :(
#tol = 0
#NbIter = 500
#NbIter_hals = 150
#NbIter_inner = 10
#Nb_seeds = 1
#pert_sigma = 1
#use_gt = 0
#epsilon=1e-8

#df = pd.DataFrame()

## Now compute the NMF of M
#for s in range(Nb_seeds):

    ## print
    #print("Loop number is", s)

    ## Perturbing the initialization for randomization
    #Wini = use_gt*Wref + pert_sigma*np.random.rand(m, rank)
    #Hini = use_gt*Href + pert_sigma*np.random.rand(rank, n)

    #print('Running Lee and Seung NMF')
    #error0, W0, H0, toc0 = nmf_f.NMF_Lee_Seung(M,  Wini, Hini, NbIter, NbIter_inner,tol=tol, legacy=False, epsilon=epsilon)
    #print('Running our proposed NeNMF')
    #error1, W1, H1, toc1  = nmf_f.NeNMF_optimMajo(M, Wini, Hini, tol=tol, itermax=NbIter, nb_inner=NbIter_inner, epsilon=epsilon)
    #print('Running Proximal Gradient Descent')
    ##error1, W1, H1, toc1 = nmf_f.NMF_proposed_Frobenius(V, Wini, Hini, NbIter, NbIter_inner, tol=tol)
    #error2, W2, H2, toc2  = nmf_f.Grad_descent(M , Wini, Hini, NbIter, NbIter_inner, tol=tol, epsilon=epsilon)
    #print('Running NeNMF')
    #error3, W3, H3, toc3  = nmf_f.NeNMF(M, Wini, Hini, tol=tol, nb_inner=NbIter_inner, itermax=NbIter, epsilon=epsilon)
    ## Fewer max iter cause too slow
    #print('Running HALS')
    #W4, H4, error4, toc4 = nn_fac.nmf.nmf(M, rank, init="custom", U_0=np.copy(Wini), V_0=np.copy(Hini), n_iter_max=NbIter_hals, tol=tol, update_rule='hals',beta=2, return_costs=True, NbIter_inner=NbIter_inner)
    
    #dic= {
            #"batch_size": 5, # number of algorithms in each comparison run
            #"method": ["NMF_LeeSeung (modern)",  "NeNMF_optimMajorant", "PGD", "NeNMF", "HALS"],
            #"m": m,
            #"n": n,
            #"r": rank,
            #"seed_idx": s,
            #"epsilon": epsilon,
            #"pert_sigma": pert_sigma,
            #"use_gt": use_gt,
            #"NbIter": NbIter,
            #"NbIter_inner": NbIter_inner,
            #"NbIter_hals": NbIter_hals,
            #"tol": tol,
            #"final_error": [error0[-1], error1[-1], error2[-1], error3[-1], error4[-1]],
            #"total_time": [toc0[-1], toc1[-1], toc2[-1], toc3[-1], toc4[-1]],
            #"full_error": [error0,error1,error2,error3,error4],
            #"full_time": [toc0,toc1,toc2,toc3,toc4],
            #"NbIterStop": [len(error0),len(error1),len(error2),len(error3),len(error4)]
        #}

    #df = pd.concat([df,pd.DataFrame(dic)], ignore_index=True)



## Winner at given threshold plots
#min_thresh = np.log10(error0[0])
#max_thresh = np.log10(error1[-1])
#thresh = np.logspace(min_thresh,max_thresh-1,50)
#scores_time, scores_it, timings, iterations = utils.find_best_at_all_thresh(df,thresh, 5)

#fig0 = plt.figure()
#plt.subplot(121)
#plt.semilogx(thresh, scores_time.T)
#plt.legend(["NMF_LeeSeung",  "NeNMF_optimMajorant", "PGD", "NeNMF", "HALS"])
#plt.title('How many times each algorithm reached threshold the fastest (time)')
#plt.xlabel('Rec error threshold')
#plt.ylabel('Number of faster runs')
#plt.subplot(122)
#plt.semilogx(thresh, scores_it.T)
#plt.legend(["NMF_LeeSeung",  "NeNMF_optimMajorant", "PGD", "NeNMF", "HALS"])
#plt.title('How many times each algorithm reached threshold the fastest (iters)')
#plt.xlabel('Rec error threshold')
#plt.ylabel('Number of faster runs')

## Error plots
#fig_convergence_plots = plt.figure()
#plt.semilogy(toc0,error0, label="Lee Seung NMF")
#plt.semilogy(toc1,error1, label="Phan NeNMF")
#plt.semilogy(toc2,error2, label="PGD")
#plt.semilogy(toc3,error3, label="NeNMF")
#plt.semilogy(toc4,error4, label="HALS")
#plt.legend(fontsize = 14)

#fig_convergence_plots_it = plt.figure()
#plt.semilogy(error0, label="Lee Seung NMF")
#plt.semilogy(error1, label="Phan NeNMF")
#plt.semilogy(error2, label="PGD")
#plt.semilogy(error3, label="NeNMF")
#plt.semilogy(error4, label="HALS")
#plt.legend(fontsize = 14)
#plt.show()

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
#UtM = Wref.T@M
#UtU = Wref.T@Wref
#H = nn_fac.nnls.hals_nnls_acc(UtM,UtU, in_V = np.random.rand(Wref.shape[1],M.shape[1]),delta=1e-8, nonzero=False, maxiter=100)[0]
##H = fista(UtM,UtU, tol=1e-8, n_iter_max=100)
## Postprocessing
#Mest = Wref@H
## Error computation
#print('Relative Frobenius error with randomly picked pixels, FISTA alg', np.linalg.norm(M - Mest2, 'fro'))