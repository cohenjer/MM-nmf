from importlib.resources import path
import numpy as np
from scipy.linalg import hadamard
import NMF_Frobenius as nmf_f 
import NMF_KL as nmf_kl
import matplotlib.pyplot as plt
import nn_fac
import pandas as pd
import utils
#from nn_fac.nmf import nmf as nmf_hals
from tensorly.tenalg.proximal import fista
import soundfile as sf
from scipy import signal
import plotly.express as px
# personal toolbox
from shootout.methods.runners import run_and_track
from shootout.methods.post_processors import find_best_at_all_thresh, df_to_convergence_df, error_at_time_or_it
from shootout.methods.plotters import plot_speed_comparison
'''
 W is a dictionary of 88 columns and 4097 frequency bins. Each column was obtained by performing a rank-one NMF (todo correct) on the recording of a single note in the MAPS database, on a Yamaha Disklavier with close microphones, the note was played mezzo forte and the loss was beta-divergence with beta=1.

 By performing matrix NNLS from the Power STFT Y of 30s of a (relatively simple) song in MAPS, recorded with the same piano in similar conditions and processes in the same way, we expect that $Y \approx WH$, where H are the activations of each note in the recording. A good loss to measure discrepencies here is the beta-divergence with beta=1.

 For the purpose of this toy experiment, only one song from MAPS is selected. We then perform NMF, and look at the activations as a piano roll.

 For the NMF part, we simply discard the provided templates, and estimate both the templates and the activations. Again it is best to use KL divergence. We can initialize with the provided template to get a initial dictionary.

 Input parameters:
 rank : int (default 88) number of notes to estimate in the music excerpt.
'''

#-------------------------------------------------------------------------
# Modeling/audio data

# Importing data and computing STFT using the Attack-Decay paper settings
# Read the song (you can use your own!)
the_signal, sampling_rate_local = sf.read('./data_and_scripts/MAPS_MUS-bach_847_AkPnBcht.wav')
# Using the settings of the Attack-Decay transcription paper
the_signal = the_signal[:,0] # left channel only
frequencies, time_atoms, Y = signal.stft(the_signal, fs=sampling_rate_local, nperseg=4096, nfft=8192, noverlap=4096 - 882)
time_step = time_atoms[1] #20 ms
freq_step = frequencies[1] #5.3 hz
#time_atoms = time_atoms # ds scale
# Taking the amplitude spectrogram
Y = np.abs(Y)**2
# Cutting silence, end song and high frequencies (>5300 Hz)
cutf = 500 #todo increase
cutt_in = int(1/time_step) # song beginning after 1 second
cutt_out = int(30/time_step)# 30seconds with 20ms steps #time_atoms.shape[0]
Y = Y[:cutf, cutt_in:cutt_out]
# normalization
#Y = Y/np.linalg.norm(Y, 'fro')

# -------------------- For NNLS -----------------------
# Importing a good dictionnary for the NNLS part
Wgt_attack = np.load('./data_and_scripts/attack_dict_piano_AkPnBcht_beta_1_stftAD_True_intensity_M.npy')
Wgt_decay = np.load('./data_and_scripts/decay_dict_piano_AkPnBcht_beta_1_stftAD_True_intensity_M.npy')
Wgt = np.concatenate((Wgt_attack,Wgt_decay),axis=1)
# Also cutting the dictionary
Wgt = Wgt[:cutf,:]
# -----------------------------------------------------


#------------------------------------------------------------------
# Computing the NMF to try and recover activations and templates
m, n = Y.shape
rank = 88*2 # one template per note only for speed
#Wgt = Wgt[:,:rank]

# Test: changing the data
#Htrue = sparsify(np.random.rand(rank,n),0.5)
#Y = Wgt@Htrue + 1e-3*np.random.rand(*Y.shape)

name = "audio_test_20-09-2022"

df = pd.DataFrame()


Nb_seeds=1
algs = ["Proposed_l2_delta1.8", "Proposed_l2_extrapolated", "GD_l2", "NeNMF_l2", "HALS_l2", "Lee_Sung_kl", "Proposed_KL"]
# TODO: better error message when algs dont match

@run_and_track(
    nb_seeds=Nb_seeds,
    algorithm_names=algs, 
    path_store="Results/",
    name_store=name,
)
def one_run(rank = rank,
            tol = 0,
            NbIter = 100,
            NbIter_inner = 100,
            delta=0.1,
            epsilon = 1e-8):
    # Perturbing the initialization for randomization
    Wini = Wgt + 0.1*np.random.rand(m,rank)
    Hini = np.random.rand(rank, n)

    # Frobenius algorithms
    #error0, W0, H0, toc0, cnt0 = nmf_f.NMF_Lee_Seung(Y,  Wini, Hini, NbIter, NbIter_inner,tol=tol, legacy=False, epsilon=epsilon, verbose=True, delta=delta)   
    error0, W0, H0, toc0, cnt0 = nmf_f.NMF_proposed_Frobenius(Y, Wini, Hini, NbIter, NbIter_inner, tol=tol, use_LeeS=False, delta=delta, verbose=True)
    error1, W1, H1, toc1, cnt1  = nmf_f.NeNMF_optimMajo(Y, Wini, Hini, tol=tol, itermax=NbIter, nb_inner=NbIter_inner, epsilon=epsilon, verbose=True, delta=delta)
    error2, W2, H2, toc2, cnt2  = nmf_f.Grad_descent(Y , Wini, Hini, NbIter, NbIter_inner, tol=tol, epsilon=epsilon, verbose=True, delta=delta)
    error3, W3, H3, toc3, cnt3  = nmf_f.NeNMF(Y, Wini, Hini, tol=tol, nb_inner=NbIter_inner, itermax=NbIter, epsilon=epsilon, verbose=True, delta=delta)
    W4, H4, error4, toc4, cnt4 = nn_fac.nmf.nmf(Y, rank, init="custom", U_0=np.copy(Wini), V_0=np.copy(Hini), n_iter_max=NbIter, tol=tol, update_rule='hals',beta=2, return_costs=True, NbIter_inner=NbIter_inner, verbose=True, delta=delta)

    # KL algorithms
    error5, W5, H5, toc5, cnt5 = nmf_kl.Lee_Seung_KL(Y, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=True, print_it=1)
    error6, W6, H6, toc6, cnt6 = nmf_kl.Proposed_KL(Y, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=True, print_it=1)


    return {
        "errors": [error0, error1, error2, error3, error4, error5, error6],
        "timings": [toc0,toc1,toc2,toc3,toc4,toc5,toc6],
        "loss": 5*["l2"]+2*["kl"],
            }
    

df = pd.read_pickle("Results/"+name)

# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
df_l2_conv = df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               filters={"loss":"l2"})

df_kl_conv = df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               filters={"loss":"kl"})
# ----------------------- Plot --------------------------- #
#fig_winner = plot_speed_comparison(thresh, scores_time, scores_it, legend=algs)
#fig_winner.show()

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
# Winner at given threshold plots
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
#plt.show()


#-----------------------------------------------------------------
# Results post-processing

# Normalize output
#W = W0
#H = H0
#normsW = np.sum(W,axis=0)
#W = W/normsW
##H = np.diag(1/np.max(H,1))@H
#H = np.diag(normsW)@H

## Printing W and H
#plt.figure()
#plt.subplot(121)
#plt.imshow(W[:200, :], aspect='auto')
#ticks = np.trunc(frequencies[0:200:10])
#plt.yticks(range(0,200,10), ticks.astype(int))
#plt.ylabel('Hz')
#plt.title('W learnt')
#plt.subplot(122)
#plt.imshow(Wini[:200, :], aspect='auto')
#ticks = np.trunc(frequencies[0:200:10])
#plt.yticks(range(0,200,10), ticks.astype(int))
#plt.title('Provided pre-trained W')
#plt.ylabel('Hz')
#plt.xticks(range(8),notes)

# Plotting H
#plt.figure()
#for i in range(rank):
#    plt.subplot(rank,1,i+1)
#    plt.plot(H[i,:])
#    plt.xticks([])
#    plt.yticks([])
#    if i==rank-1:
#        hop = 100
#        ticks = np.trunc(10*time_atoms[cutt_in:cutt_out:hop])/10
#        ticks_number = ticks.shape[0]
#        plt.xticks(range(0,ticks_number*hop,hop), ticks)
#    #plt.ylabel(notes[i])

# Plotting H version 2
# Thresholding the H values for activation detection, and plotting bitmap
#thres = 1e-3 # todo handfix
#H_plot = np.copy(H)
#H_plot[H_plot<thres]=0
#H_plot[H_plot>=thres]=1
#H_nnls_plot = np.copy(H_nnls)
#H_nnls_plot[H_nnls_plot<thres]=0
#H_nnls_plot[H_nnls_plot>=thres]=1
#hop = 100
#plt.figure()
#plt.subplot(121)
#plt.imshow(H_plot, aspect='auto', interpolation='none')
#plt.xticks([])
#plt.yticks([])
#ticks = np.trunc(10*time_atoms[cutt_in:cutt_out:hop])/10
#ticks_number = ticks.shape[0]
#plt.xticks(range(0,ticks_number*hop,hop), ticks)
#plt.ylabel('notes (not labeled)')
#plt.title('H with NMF')
#plt.subplot(122)
#plt.imshow(H_nnls_plot, aspect='auto', interpolation='none')
#plt.xticks([])
#plt.yticks([])
#ticks = np.trunc(10*time_atoms[cutt_in:cutt_out:hop])/10
#ticks_number = ticks.shape[0]
#plt.xticks(range(0,ticks_number*hop,hop), ticks)
#plt.title('H with nnls')
#plt.ylabel('notes in order')

## Printing Y
#plt.figure()
#plt.subplot(211)
#plt.imshow(Y[:200,:])
#yticks = np.trunc(frequencies[0:200:20])
#plt.yticks(range(0,200,20), yticks.astype(int))
#plt.ylabel('Hz')
#hop = 100
#xticks = np.trunc(10*time_atoms[cutt_in:cutt_out:hop])/10
#ticks_number = xticks.shape[0]
#plt.xticks(range(0,ticks_number*hop,hop), xticks)
#plt.xlabel('time (s)')
#plt.title('Y')
#plt.subplot(212)
#plt.imshow(np.sqrt(Y[:200,:]))
#plt.yticks(range(0,200,20), yticks.astype(int))
#plt.ylabel('Hz')
#plt.xticks(range(0,ticks_number*hop,hop), xticks)
#plt.xlabel('time (s)')
#plt.title('sqrt(Y)')


#plt.show()
