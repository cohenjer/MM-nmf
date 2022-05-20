import numpy as np
from scipy.linalg import hadamard
import NMF_Frobenius as nmf_f 
import matplotlib.pyplot as plt
import nn_fac
import pandas as pd
import utils
#from nn_fac.nmf import nmf as nmf_hals
from tensorly.tenalg.proximal import fista
import soundfile as sf
from scipy import signal
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
the_signal, sampling_rate_local = sf.read('./data_and_scripts/MAPS_MUS-bk_xmas1_ENSTDkCl.wav')
# Using the settings of the Attack-Decay transcription paper
the_signal = the_signal[:,0] + the_signal[:,1] # summing left and right channels
frequencies, time_atoms, Y = signal.stft(the_signal, fs=sampling_rate_local, nperseg=4096, nfft=8192, noverlap=4096 - 882)
time_step = time_atoms[1] #20 ms
freq_step = frequencies[1] #5.3 hz
#time_atoms = time_atoms # ds scale
# Taking the power spectrogram
Y = np.abs(Y)**2
# adding some constant noise for avoiding zeros
#Y = Y+1e-8
# Cutting silence, end song and high frequencies (>10600 Hz)
cutf = 2000
cutt_in = 0 # song beginning
cutt_out = int(30/time_step)# 30seconds with 20ms steps #time_atoms.shape[0]
Y = Y[:cutf, cutt_in:cutt_out]

# -------------------- For NNLS -----------------------
# Importing a good dictionnary for the NNLS part
Wgt = np.load('./data_and_scripts/attack_dict_piano_ENSTDkCl_beta_1_stftAD_True_intensity_M.npy')
# Also cutting the dictionary
Wgt = Wgt[:cutf,:]
# -----------------------------------------------------


#------------------------------------------------------------------
# NNLS with fixed dictionary

H_nnls = fista(Wgt.T@Y, Wgt.T@Wgt, tol=1e-16, n_iter_max=1000)
Hgt = np.copy(H_nnls)


#------------------------------------------------------------------
# Computing the NMF to try and recover activations and templates
m, n = Y.shape
rank = 88
beta = 2 # beta=1 data is big :(
tol = 0
NbIter = 500
NbIter_hals = 150
NbIter_inner = 10
Nb_seeds = 5
pert_sigma = 0.1
epsilon = 1e-8
use_gt = 0

df = pd.DataFrame()

Wgt = Wgt[:,:rank]
Hgt = Hgt[:rank,:]

for s in range(Nb_seeds):

    # print
    print("Loop number is", s)

    # Perturbing the initialization for randomization
    Wini = use_gt*Wgt + pert_sigma*np.random.rand(m, rank)
    Hini = use_gt*Hgt + pert_sigma*np.random.rand(rank, n)

    print('Running Lee and Seung NMF')
    error0, W0, H0, toc0 = nmf_f.NMF_Lee_Seung(Y,  Wini, Hini, NbIter, NbIter_inner,tol=tol, legacy=False, epsilon=epsilon)
    print('Running our proposed NeNMF')
    error1, W1, H1, toc1  = nmf_f.NeNMF_optimMajo(Y, Wini, Hini, tol=tol, itermax=NbIter, nb_inner=NbIter_inner, epsilon=epsilon)
    print('Running proposed MU')
    # todo
    #error5, W5, H5, toc5 = nmf_f.NMF_proposed_Frobenius(Y, Wini, Hini, NbIter, NbIter_inner, tol=tol)
    print('Running Proximal Gradient Descent')
    error2, W2, H2, toc2  = nmf_f.Grad_descent(Y , Wini, Hini, NbIter, NbIter_inner, tol=tol, epsilon=epsilon)
    print('Running NeNMF')
    error3, W3, H3, toc3  = nmf_f.NeNMF(Y, Wini, Hini, tol=tol, nb_inner=NbIter_inner, itermax=NbIter, epsilon=epsilon)
    # Fewer max iter cause too slow
    print('Running HALS')
    W4, H4, error4, toc4 = nn_fac.nmf.nmf(Y, rank, init="custom", U_0=np.copy(Wini), V_0=np.copy(Hini), n_iter_max=NbIter_hals, tol=tol, update_rule='hals',beta=2, return_costs=True, NbIter_inner=NbIter_inner)
    
    dic= {
            "batch_size": 5, # number of algorithms in each comparison run
            "method": ["NMF_LeeSeung (modern)",  "NeNMF_optimMajorant", "PGD", "NeNMF", "HALS"],
            "m": m,
            "n": n,
            "r": rank,
            "seed_idx": s,
            "use_gt": use_gt,
            "epsilon": epsilon,
            "pert_sigma": pert_sigma,
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
plt.show()


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
