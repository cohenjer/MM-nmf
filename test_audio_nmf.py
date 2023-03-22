from importlib.resources import path
import numpy as np
from scipy.linalg import hadamard
import NMF_Frobenius as nmf_f 
import NMF_KL as nmf_kl
import matplotlib.pyplot as plt
import nn_fac
import pandas as pd
import soundfile as sf
from scipy import signal
import plotly.express as px
# personal toolbox
from shootout.methods.runners import run_and_track
from shootout.methods.post_processors import df_to_convergence_df
import sys
import plotly.io as pio
pio.kaleido.scope.mathjax = None
pio.templates.default= "plotly_white"
'''
 W is a dictionary of 88 columns and 4097 frequency bins. Each column was obtained by performing a rank-one NMF (todo correct) on the recording of a single note in the MAPS database, on a Yamaha Disklavier with close microphones, the note was played mezzo forte and the loss was beta-divergence with beta=1.

 By performing matrix NNLS from the Power STFT Y of 30s of a (relatively simple) song in MAPS, recorded with the same piano in similar conditions and processes in the same way, we expect that $Y \approx WH$, where H are the activations of each note in the recording. A good loss to measure discrepencies here is the beta-divergence with beta=1.

 For the purpose of this toy experiment, only one song from MAPS is selected. We then perform NMF, and look at the activations as a piano roll.

 For the NMF part, we simply discard the provided templates, and estimate both the templates and the activations. Again it is best to use KL divergence. We can initialize with the provided template to get a initial dictionary.
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
cutf = 1000 
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

name = "audio_test_02-14-2023"

df = pd.DataFrame()


if len(sys.argv)==1:
    nb_seeds = 0 #no run
else:
    nb_seeds = int(sys.argv[1])  # Change this to >0 to run experiments
algs = ["fastMU_Fro", "fastMU_Fro_ex", "GD_Fro", "NeNMF_Fro", "MU_Fro", "HALS", "MU_KL", "fastMU_KL", "fastMU_KL_approx"]
# TODO: better error message when algs dont match

@run_and_track(
    nb_seeds=nb_seeds,
    algorithm_names=algs, 
    path_store="Results/",
    name_store=name,
)
def one_run(rank = rank,
            tol = 0,
            NbIter = 100,
            NbIter_inner = 100,
            delta=0.1,
            verbose=True,
            epsilon = 1e-8):
    # Perturbing the initialization for randomization
    Wini = Wgt + 0.1*np.random.rand(m,rank)
    Hini = np.random.rand(rank, n)

    # Frobenius algorithms
    error0, W0, H0, toc0, cnt0 = nmf_f.NMF_proposed_Frobenius(Y, Wini, Hini, NbIter, NbIter_inner, tol=tol, use_LeeS=False, delta=delta, verbose=verbose, gamma=1.9)
    #error1, W1, H1, toc1, cnt1 = nmf_f.NMF_proposed_Frobenius(Y, Wini, Hini, NbIter, NbIter_inner, tol=tol, use_LeeS=True, delta=delta, verbose=verbose, gamma=1)
    error2, W2, H2, toc2, cnt2  = nmf_f.NeNMF_optimMajo(Y, Wini, Hini, tol=tol, itermax=NbIter, nb_inner=NbIter_inner, epsilon=epsilon, verbose=verbose, delta=delta, gamma=1)
    error3, W3, H3, toc3, cnt3  = nmf_f.Grad_descent(Y , Wini, Hini, NbIter, NbIter_inner, tol=tol, epsilon=epsilon, verbose=verbose, delta=delta)
    error4, W4, H4, toc4, cnt4  = nmf_f.NeNMF(Y, Wini, Hini, tol=tol, nb_inner=NbIter_inner, itermax=NbIter, epsilon=epsilon, verbose=verbose, delta=delta)
    error5, W5, H5, toc5, cnt5 = nmf_f.NMF_Lee_Seung(Y,  Wgt, Hini, NbIter, NbIter_inner, tol=tol, legacy=False, verbose=verbose, delta=delta, epsilon=epsilon)
    W6, H6, error6, toc6, cnt6 = nn_fac.nmf.nmf(Y, rank, init="custom", U_0=np.copy(Wini), V_0=np.copy(Hini), n_iter_max=NbIter, tol=tol, update_rule='hals',beta=2, return_costs=True, NbIter_inner=NbIter_inner, verbose=verbose, delta=delta)

    # KL algorithms
    error7, W7, H7, toc7, cnt7 = nmf_kl.Lee_Seung_KL(Y, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, epsilon=epsilon)
    error8, W8, H8, toc8, cnt8 = nmf_kl.Proposed_KL(Y, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, use_LeeS=False, gamma=1.9, epsilon=epsilon, true_hessian=True)
    error9, W9, H9, toc9, cnt9 = nmf_kl.Proposed_KL(Y, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, use_LeeS=False, gamma=1.9, epsilon=epsilon, true_hessian=False)


    return {
        "errors": [error0, error2, error3, error4, error5, error6, error7, error8, error9],
        "timings": [toc0,toc2,toc3,toc4,toc5,toc6, toc7, toc8, toc9],
        "loss": 6*["l2"]+3*["kl"],
            }
    

df = pd.read_pickle("Results/"+name)

# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
df_l2_conv = df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               filters={"loss":"l2"})

df_kl_conv = df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               filters={"loss":"kl"})
# ----------------------- Plot --------------------------- #

# Convergence plots with all runs
pxfig = px.line(df_l2_conv, line_group="groups", x="timings", y= "errors", color='algorithm', 
            line_dash='algorithm',
            log_y=True)

# Final touch
pxfig.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfig.update_layout(
    title_text = "NMF",
    font_size = 12,
    width=450*1.62/2, # in px
    height=450,
    xaxis=dict(range=[0,10], title_text="Time (s)"),
    yaxis=dict(range=np.log10([5e-11,1e-7]), title_text="Fit")
)

pxfig.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfig.update_yaxes(
    matches=None,
    showticklabels=True
)

pxfig.write_image("Results/"+name+"_fro.pdf")
pxfig.write_image("Results/"+name+"_fro.pdf")
pxfig.show()


pxfig2 = px.line(df_kl_conv, line_group="groups", x="timings", y= "errors", color='algorithm',
            line_dash='algorithm',
            log_y=True)

# Final touch
pxfig2.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfig2.update_layout(
    title_text = "NMF",
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