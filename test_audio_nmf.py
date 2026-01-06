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
import shootout.methods.post_processors as pp
import sys
from utils import opt_scaling
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
the_signal, sampling_rate_local = sf.read('./data_and_scripts/MAPS_MUS-bach_846_AkPnBcht.wav')
# Using the settings of the Attack-Decay transcription paper
the_signal = the_signal[:,0] # left channel only
frequencies, time_atoms, Y = signal.stft(the_signal, fs=sampling_rate_local, nperseg=4096, nfft=8192, noverlap=4096 - 882)
time_step = time_atoms[1] #20 ms
freq_step = frequencies[1] #5.3 hz
# Taking the amplitude spectrogram
Y = np.abs(Y)
# Cutting silence, end song and high frequencies (>5300 Hz)
cutf = 1000 
cutt_in = int(1/time_step) # song beginning after 1 second
cutt_out = int(30/time_step)# 30seconds with 20ms steps #time_atoms.shape[0]
Y = Y[:cutf, cutt_in:cutt_out]
# normalization
Y = Y/np.max(Y)  # does not change much

df = pd.DataFrame()

if len(sys.argv)==1 or int(sys.argv[1])==0:
    seeds = []  # no run
    skip=True
else:
    seeds = list(np.arange(int(sys.argv[1])))
    skip=False

# TODO More iterations? show all runs ?
variables = {
    "NbIter": 400,
    "NbIter_SN": 100,  #un peu long
    "NbIter_inner": 10,
    "NbIter_inner_SN": 5,
    "delta": 0,
    "epsilon": 1e-16,
    "rank": [2, 11, 23, 45],
    "seed": seeds,
    "sigma": 0.1,
    "cutf": cutf,
    "tol": 0
}

#name = "audio_test_01-06-2024"
name = "audio_18-04-2025"

#algs = ["AmSOM", "APGD", "NeNMF", "AMU", "HALS", "AMU_kl", "AmSOM_kl", "AMUSOM_kl", "ASN CCD"]
algs = ["AMU", "AmSOM", "AMUSOM", "ASN CCD"]
# TODO: better error message when algs dont match

@run_and_track(
    algorithm_names=algs, 
    path_store="Results/",
    name_store=name,
    skip=skip,
    **variables
)
def one_run(rank=12,
            tol = 0,
            seed = 1,
            sigma = 0.1,
            NbIter = 100,
            NbIter_SN = 50,
            NbIter_inner = 100,
            NbIter_inner_SN = 50,
            cutf=1000,
            delta=0.1,
            verbose=True,
            epsilon = 1e-8):
    # Importing a good dictionnary for the NNLS part
    Wgt = np.load('./data_and_scripts/attack_dict_piano_AkPnBcht_beta_1_stftAD_True_intensity_M.npy')
    Wgt = Wgt[:cutf, :]

    Wgt = Wgt[:,27:(27+rank)] # octaves in the middle
    # Normalization by max
    Wgt = Wgt/np.max(Wgt, axis=0)

    #------------------------------------------------------------------
    # Computing the NMF to try and recover activations and templates
    m, n = Y.shape
    
    # Perturbing the initialization for randomization
    rng = np.random.RandomState(seed+20)
    Wini = Wgt + sigma*rng.rand(m, rank)
    Hini = rng.rand(rank, n)
    lamb = opt_scaling(Y, Wini@Hini)
    Hini = lamb*Hini
    _, Wini, Hini, _, _ = nmf_kl.Lee_Seung_KL(Y, Wini, Hini, NbIter=1, nb_inner=NbIter_inner, tol=0, verbose=verbose, epsilon=epsilon, print_it=1)

    # KL algorithms
    error7, W7, H7, toc7, cnt7 = nmf_kl.Lee_Seung_KL(Y, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, epsilon=epsilon, print_it=20)
    error8, W8, H8, toc8, cnt8 = nmf_kl.Proposed_KL(Y, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, gamma=1.9, epsilon=epsilon, print_it=20)
    error9, W9, H9, toc9, cnt9 = nmf_kl.Proposed_KL(Y, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, gamma=1.9, epsilon=epsilon, method="AMUSOM", print_it=20)
    error10, W10, H10, toc10, cnt10 = nmf_kl.ScalarNewton(Y, Wini, Hini, NbIter=NbIter_SN, nb_inner=NbIter_inner_SN, tol=tol, verbose=verbose,  epsilon=epsilon, method="CCD", print_it=5)


    return {
        "errors": [error7, error8, error9, error10],
        "timings": [toc7, toc8, toc9, toc10],
        #"loss": 5*["l2"]+4*["kl"],
            }
    

df = pd.read_pickle("Results/"+name)

# Remove extrapolation
#df = df[df["algorithm"] != "fastMU_Fro_ex"]

ovars_iterp = ["algorithm", "rank"]
df = pp.interpolate_time_and_error(df, npoints = 100, adaptive_grid=True, groups=ovars_iterp)

# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
#df_l2_conv = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               #filters={"loss":"l2"}, err_name="errors_interp", time_name="timings_interp")
#df_l2_conv = df_l2_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
#df_l2_conv_it = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               #filters={"loss":"l2"})

df_kl_conv = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=["rank"],
                               err_name="errors_interp", time_name="timings_interp")
df_kl_conv = df_kl_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
df_kl_conv_it = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=["rank"])

#df_l2_conv_median_time = pp.median_convergence_plot(df_l2_conv, type_x="timings")
df_kl_conv_median_time = pp.median_convergence_plot(df_kl_conv, type_x="timings")
#df_l2_conv_median_it = pp.median_convergence_plot(df_l2_conv_it)
df_kl_conv_median_it = pp.median_convergence_plot(df_kl_conv_it)
# ----------------------- Plot --------------------------- #
pxfig2 = px.line(df_kl_conv_median_time, #line_group="groups", 
                 x="timings", y= "errors", color='algorithm',
                 line_dash='algorithm',
                 facet_col="rank",
                 facet_col_wrap=2,
                 facet_col_spacing=0.07,
                 facet_row_spacing=0.17,
                 log_y=True,
                 log_x=True,
                 )

pxfig2it = px.line(df_kl_conv_median_it, 
            x="it", 
            y= "errors", 
            color='algorithm',
            line_dash='algorithm',
            log_y=True,
            log_x=True,
            #line_group="groups",
            facet_col="rank",
            facet_col_wrap=2,
            facet_col_spacing=0.07,
            facet_row_spacing=0.17,
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
    title_text = "NMF",
    font_size = 10,
    width=450, # in px
    height=350,
    xaxis1=dict(range=np.log10([2, 100]), title_text="Time (s)"),
    xaxis2=dict(range=np.log10([2, 100]), title_text="Time (s)"),
    xaxis3=dict(range=np.log10([0, 20])),
    xaxis4=dict(range=np.log10([4, 40])),
    yaxis1=dict(range=np.log10([450, 550]), title_text="Loss"),
    yaxis2=dict(range=np.log10([200, 500])),
    yaxis3=dict(range=np.log10([5500, 9000]), title_text="Loss"),
    yaxis4=dict(range=np.log10([1270, 1320])),
)

pxfig2.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfig2.update_yaxes(
    matches=None,
    showticklabels=True
)

pxfig2it.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfig2it.update_layout(
    title_text = "NMF",
    font_size = 10,
    #width=450*1.62/2, # in px
    #height=450,
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
#pxfig = px.line(df_l2_conv_median_time, #line_group="groups",
                #x="timings", y= "errors", color='algorithm', 
            #line_dash='algorithm',
            #log_y=True)

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
    #title_text = "NMF",
    #font_size = 12,
    #width=450*1.62/2, # in px
    #height=450,
    ##xaxis=dict(range=[0,10], title_text="Time (s)"),
    ##yaxis=dict(range=np.log10([5e-11,1e-7]), title_text="Fit")
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
    #title_text = "NMF",
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

