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
import sys
import plotly.io as pio
pio.kaleido.scope.mathjax = None
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

algs = ["fastMU_Fro", "fastMU_Fro_ex", "GD_Fro", "NeNMF_Fro", "MU_Fro", "HALS", "MU_KL", "fastMU_KL", "fastMU_KL_approx"]
name = "hsi_nmf_test_02_14_2023"
if len(sys.argv)==1:
    nb_seeds = 0 #no run
else:
    nb_seeds = int(sys.argv[1])  # Change this to >0 to run experiments
@run_and_track(
    nb_seeds=nb_seeds,
    algorithm_names=algs, 
    path_store="Results/",
    name_store=name,
)
def one_run(rank = 6,
            NbIter = 200,
            NbIter_inner = 100,
            delta=0.1,
            epsilon = 1e-8,
            tol=0,
            verbose=False,
            print_it=100,
            seed = 1,
            ):
    # Seeding
    rng = np.random.RandomState(seed+20)
    # Init
    Wini = Wref + 0.1*np.random.rand(m,rank)
    Hini = rng.rand(rank, n)

    # Frobenius algorithms
    error0, W0, H0, toc0, cnt0 = nmf_f.NMF_Lee_Seung(M,  Wini, Hini, NbIter, NbIter_inner,tol=tol, legacy=False, epsilon=epsilon, verbose=verbose, delta=delta, print_it=print_it)   
    error1, W1, H1, toc1, cnt1 = nmf_f.NMF_proposed_Frobenius(M, Wini, Hini, NbIter, NbIter_inner, tol=tol, use_LeeS=False, delta=delta, verbose=verbose, print_it=print_it, gamma=1.9)
    #error2, W2, H2, toc2, cnt2 = nmf_f.NMF_proposed_Frobenius(M, Wini, Hini, NbIter, NbIter_inner, tol=tol, use_LeeS=True, delta=delta, verbose=verbose, print_it=print_it, gamma=1)
    error3, W3, H3, toc3, cnt3  = nmf_f.NeNMF_optimMajo(M, Wini, Hini, tol=tol, itermax=NbIter, nb_inner=NbIter_inner, epsilon=epsilon, verbose=verbose, delta=delta, print_it=print_it, gamma=1)
    error4, W4, H4, toc4, cnt4  = nmf_f.Grad_descent(M , Wini, Hini, NbIter, NbIter_inner, tol=tol, epsilon=epsilon, verbose=verbose, delta=delta, print_it=print_it)
    error5, W5, H5, toc5, cnt5  = nmf_f.NeNMF(M, Wini, Hini, tol=tol, nb_inner=NbIter_inner, itermax=NbIter, epsilon=epsilon, verbose=verbose, delta=delta, print_it=print_it)
    W6, H6, error6, toc6, cnt6 = nn_fac.nmf.nmf(M, rank, init="custom", U_0=np.copy(Wini), V_0=np.copy(Hini), n_iter_max=NbIter, tol=tol, update_rule='hals',beta=2, return_costs=True, NbIter_inner=NbIter_inner, verbose=verbose, delta=delta)

    # KL algorithms
    error7, W7, H7, toc7, cnt7 = nmf_kl.Lee_Seung_KL(M, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=print_it)
    error8, W8, H8, toc8, cnt8 = nmf_kl.Proposed_KL(M, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=print_it, use_LeeS=False, gamma=1.9, true_hessian=True)
    error9, W9, H9, toc9, cnt9 = nmf_kl.Proposed_KL(M, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=print_it, use_LeeS=False, gamma=1.9, true_hessian=False)

    return {
        "errors": [error1,error3,error4, error5, error0, error6, error7, error8, error9],
        "timings": [toc1,toc3,toc4,toc5,toc0,toc6,toc7,toc8,toc9],
        "loss": 6*["l2"]+3*["kl"],
    }

df = pd.read_pickle("Results/"+name)

# Making a convergence plot dataframe
df_l2_conv = df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               filters={"loss":"l2"})

df_kl_conv = df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                               filters={"loss":"kl"})
# ----------------------- Plot --------------------------- #
# Convergence plots with all runs
pxfig = px.line(df_l2_conv,
            line_group="groups",
            x="timings",
            y="errors",
            color='algorithm',
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
    xaxis=dict(range=[0,50], title_text="Time (s)"),
    yaxis=dict(range=np.log10([0.00145,0.0020]), title_text="Fit")
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

pxfig2 = px.line(df_kl_conv,
            line_group="groups",
            x="timings",
            y= "errors",
            color='algorithm',
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