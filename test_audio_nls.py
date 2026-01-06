import numpy as np
import NLS_Frobenius as nls_f 
import NLS_KL as nls_kl
import nn_fac
import pandas as pd
import soundfile as sf
from scipy import signal
import plotly.express as px
# personal toolbox
from shootout.methods.runners import run_and_track
import shootout.methods.post_processors as pp
import time
import sys
import plotly.io as pio
from utils import opt_scaling, nearest_neighbour_H
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
the_signal, sampling_rate_local = sf.read('./data_and_scripts/MAPS_MUS-bach_846_AkPnBcht.wav')
# Using the settings of the Attack-Decay transcription paper
the_signal = the_signal[:,0] # left channel only
frequencies, time_atoms, Y = signal.stft(the_signal, fs=sampling_rate_local, nperseg=4096, nfft=8192, noverlap=4096 - 882)
time_step = time_atoms[1] #20 ms
freq_step = frequencies[1] #5.3 hz
#time_atoms = time_atoms # ds scale
# Taking the amplitude spectrogram
Y = np.abs(Y)
# Cutting silence, end song and high frequencies (>5300 Hz)
cutf = 1000
cutt_in = int(1/time_step) # song beginning after 1 second
cutt_out = int(30/time_step)# 30seconds with 20ms steps #time_atoms.shape[0]
Y = Y[:cutf, cutt_in:cutt_out]
# normalization
Y = Y/np.max(Y)
# -------------------- For NNLS -----------------------
# Importing a good dictionnary for the NNLS part
#Wgt_attack = np.load('./data_and_scripts/attack_dict_piano_AkPnBcht_beta_1_stftAD_True_intensity_M.npy')
#Wgt_decay = np.load('./data_and_scripts/decay_dict_piano_AkPnBcht_beta_1_stftAD_True_intensity_M.npy')
#Wgt = np.concatenate((Wgt_attack,Wgt_decay),axis=1)

# Shootout config
#name = "audio_nls_test_rank"
name = "audio_nls_18-04-2025"

if len(sys.argv)==1 or int(sys.argv[1])==0:
    seeds = [] #no run
    skip = True
else:
    seeds = list(np.arange(int(sys.argv[1])))
    skip = False

variables = {
    "NbIter" : 50,
    "NbIter_SN" : 10,
    "delta" : 0,
    "epsilon" : 1e-16,
    "rank": [2, 11, 23, 45],
    "cutf": cutf,
    "seed" : seeds
}

df = pd.DataFrame()

#algs = ["mSOM", "PGD", "NeNMF", "MU", "HALS", "MU_kl", "mSOM_kl", "MUSOM_kl", "SN CCD"]
algs = ["MU", "mSOM", "MUSOM", "SN CCD"]
# TODO: trueMU for Fro, init for Fro

@run_and_track(
    algorithm_names=algs, 
    path_store="Results/",
    name_store=name,
    skip=skip,
    **variables
)
def one_run(NbIter = 100,
            NbIter_SN = 30,
            delta=0, # NLS test, no early stopping
            epsilon = 1e-8,
            rank = 12,
            seed=1, # will actually be seed idx from run and track
            cutf=1000,
            verbose=True,
            ):
    # -----------------------------------------------------
    # Loading the attack dictionary only
    Wgt = np.load('./data_and_scripts/attack_dict_piano_AkPnBcht_beta_1_stftAD_True_intensity_M.npy')
    # Also cutting the dictionary
    Wgt = Wgt[:cutf,:]
    # TODO Performance of mSOM decrease wrt MU as the rank increase, I can't explain why and don't observe it in other simulations or xps.
    #Wgt = Wgt[:, 15:73]  # 4 octaves in the middle (12x4)
    Wgt = Wgt[:, 27:(27+rank)]  # octaves in the middle starting from C2, 2 octaves from lowest piano key 
    # Normalization by max
    Wgt = Wgt/np.max(Wgt,axis=0)

    #------------------------------------------------------------------
    # Computing the NMF to try and recover activations and templates
    m, n = Y.shape
    
    # Seeding
    rng = np.random.RandomState(seed+20)
    # Perturbing the initialization for randomization
    Hini = rng.rand(rank, n)
    lamb = opt_scaling(Y, Wgt@Hini)
    Hini = lamb*Hini

    # KL algorithms
    # Trying to use few iterations of MU to start fastMU
#    error7, H7, toc7 = nls_kl.Lee_Seung_KL(Y,  Wgt, Hini, NbIter=nit_mu, delta=delta, verbose=verbose, epsilon=epsilon)
#    error71, H71, toc71, = nls_kl.Proposed_KL(Y, Wgt, H7, NbIter=NbIter-nit_mu, use_LeeS=False, delta=delta, verbose=verbose, gamma=1.9)
#    error7 = error7[:-1]+error71[(nit_mu-1):] # use error from Proposed
#    toc7 = toc7 + [toc7[-1] +  i for i in toc71[nit_mu:]]
    error8, H8, toc8 = nls_kl.Lee_Seung_KL(Y, Wgt, Hini, NbIter=NbIter, verbose=verbose, delta=delta, epsilon=epsilon)
    error9, H9, toc9 = nls_kl.Proposed_KL(Y, Wgt, Hini, NbIter=NbIter, verbose=verbose, delta=delta, gamma=1.9, epsilon=epsilon)
    error10, H10, toc10 = nls_kl.Proposed_KL(Y, Wgt, Hini, NbIter=NbIter, verbose=verbose, delta=delta, gamma=1.9, epsilon=epsilon, method="MUSOM")
    error11, H11, toc11 = nls_kl.ScalarNewton(Y, Wgt, Hini, NbIter=NbIter_SN, verbose=verbose, delta=delta, epsilon=epsilon, method="CCD", print_it=10)


    # Tracking issues
    #import matplotlib.pyplot as plt
    #plt.subplot(221)
    #plt.imshow(H8[:,:200])
    #plt.subplot(222)
    #plt.plot(np.sum(H8,axis=1))
    #plt.subplot(223)
    #plt.imshow(H9[:,:200])
    #plt.subplot(224)
    #plt.plot(np.sum(H9,axis=1))
    #print(np.sum(H9,axis=1))
    #plt.show()

    return {
        "errors": [error8, error9, error10, error11],
        "timings": [toc8, toc9, toc10, toc11],
        #"loss": 5*["l2"]+4*["kl"],
            }
    
df = pd.read_pickle("Results/"+name)

# Remove extrapolation
#df = df[df["algorithm"] != "fastMU_Fro_ex"]

# no need for median plots here, only 3 runs  (too costly)

## Using shootout for plotting and postprocessing
#min_thresh = 0
#max_thresh = -10
#thresh = np.logspace(min_thresh,max_thresh,50)
#scores_time, scores_it, timings, iterations = find_best_at_all_thresh(df,thresh, Nb_seeds)

# Interpolating
ovars_iterp = ["algorithm", "rank"]
df = pp.interpolate_time_and_error(df, npoints = 50, adaptive_grid=True, groups=ovars_iterp)

# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
#df_l2_conv = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               #filters={"loss":"l2"}, err_name="errors_interp", time_name="timings_interp")
#df_l2_conv = df_l2_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
#df_l2_conv_it = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               #filters={"loss":"l2"})

other_names=["rank"]
df_kl_conv = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=other_names,
                               err_name="errors_interp", time_name="timings_interp")
df_kl_conv_it = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=other_names)
df_kl_conv = df_kl_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
# ----------------------- Plot --------------------------- #
#fig_winner = plot_speed_comparison(thresh, scores_time, scores_it, legend=algs)
#fig_winner.show()

# Median plots
#df_l2_conv_median_time = pp.median_convergence_plot(df_l2_conv, type_x="timings")
df_kl_conv_median_time = pp.median_convergence_plot(df_kl_conv, type_x="timings")
#df_l2_conv_median_it = pp.median_convergence_plot(df_l2_conv_it)
df_kl_conv_median_it = pp.median_convergence_plot(df_kl_conv_it)

pxfig2 = px.line(df_kl_conv_median_time, 
            x="timings", 
            y= "errors", 
            color='algorithm',
            line_dash='algorithm',
            facet_col="rank",
            facet_col_wrap=2,
            log_y=True,
            log_x=True,
            facet_col_spacing=0.07,
            facet_row_spacing=0.17,
            #error_y="q_errors_p", 
            #error_y_minus="q_errors_m", 
)
pxfig2it = px.line(df_kl_conv_it,  #_median_it, 
            x="it", 
            y= "errors", 
            color='algorithm',
            line_dash='algorithm',
            log_y=True,
            line_group="groups",
            facet_col="rank",
            facet_col_wrap=2,
            facet_col_spacing=0.07,
            facet_row_spacing=0.17
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
    font_size = 10,
    width=450, # in px
    height=350,
    xaxis1=dict(title_text="Time (s)"),
    xaxis2=dict(title_text="Time (s)"),
    yaxis1=dict(title_text="Loss"),
    yaxis3=dict(title_text="Loss")
)

pxfig2.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfig2.update_yaxes(
    matches=None,
    showticklabels=True
)
# Final touch
pxfig2it.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfig2it.update_layout(
    title_text = "NLS",
    font_size = 10,
    width=450, # in px
    height=350,
    #xaxis=dict(range=[0,4],title_text="Time (s)"),
    #yaxis=dict(title_text="Fit")
)

pxfig2it.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfig2it.update_yaxes(
    matches=None,
    showticklabels=True
)
pxfig2.write_image("Results/"+name+"_kl.pdf")
pxfig2it.write_image("Results/"+name+"_kl_it.pdf")
pxfig2.show()
pxfig2it.show()
# Convergence plots with all runs
#pxfig = px.line(df_l2_conv_median_time, 
            #x="timings", 
            #y= "errors", 
            #color='algorithm',
            #line_dash='algorithm',
            #log_y=True,
            ##error_y="q_errors_p", 
            ##error_y_minus="q_errors_m", 
#)

#pxfigit = px.line(df_l2_conv_median_it, 
            #x="it", 
            #y= "errors", 
            #color='algorithm',
            #line_dash='algorithm',
            #log_y=True,
            ##error_y="q_errors_p", 
            ##error_y_minus="q_errors_m", 
#)

## Final touch
#pxfig.update_traces(
    #selector=dict(),
    #line_width=2.5,
    ##error_y_thickness = 0.3,
#)

#pxfig.update_layout(
    #title_text = "NLS",
    #font_size = 12,
    #width=450*1.62/2, # in px
    #height=450,
    ##xaxis=dict(range=[0,0.5], title_text="Time (s)"),
    ##yaxis=dict(range=np.log10([2e-7,7e-7]), title_text="Fit")
#)

#pxfig.update_xaxes(
    #matches = None,
    #showticklabels = True
#)
#pxfig.update_yaxes(
    #matches=None,
    #showticklabels=True
#)
#pxfigit.update_traces(
    #selector=dict(),
    #line_width=2.5,
    ##error_y_thickness = 0.3,
#)

#pxfigit.update_layout(
    #title_text = "NLS",
    #font_size = 12,
    #width=450*1.62/2, # in px
    #height=450,
    ##xaxis=dict(range=[0,0.5], title_text="Time (s)"),
    ##yaxis=dict(range=np.log10([2e-7,7e-7]), title_text="Fit")
#)

#pxfigit.update_xaxes(
    #matches = None,
    #showticklabels = True
#)
#pxfigit.update_yaxes(
    #matches=None,
    #showticklabels=True
#)



#pxfig.write_image("Results/"+name+"_fro.pdf")
#pxfig.write_image("Results/"+name+"_fro.pdf")
#pxfigit.write_image("Results/"+name+"_fro_it.pdf")
#pxfig.show()
#pxfigit.show()




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
