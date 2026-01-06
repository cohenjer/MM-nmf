import numpy as np
import NMF_Frobenius as nmf_f 
import NMF_KL as nmf_kl
import plotly.express as px
import nn_fac
import pandas as pd
import scipy.io
from shootout.methods.runners import run_and_track
from shootout.methods import post_processors as pp
from shootout.methods.post_processors import df_to_convergence_df, interpolate_time_and_error, median_convergence_plot
import sys
import plotly.io as pio
from utils import opt_scaling_fro, nearest_neighbour_H
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
if len(sys.argv)==1 or int(sys.argv[1])==0:
    seeds = [] #no run
    skip=True
else:
    seeds = list(np.arange(int(sys.argv[1])))
    skip=False

variables = {
    "NbIter" : 400,
    "NbIter_inner" : 10,
    "delta" : 0,
    "epsilon" : 1e-16,
    "seed" : seeds,
    "tol" : 0,
}

#algs = ["fastMU_Fro", "fastMU_Fro_ex", "GD_Fro", "NeNMF_Fro", "MU_Fro", "HALS", "MU_KL", "fastMU_KL", "trueMU"]
algs = ["AMU", "AmSOM", "AMUSOM", "APGD", "NeNMF", "AHALS"]
name = "hsi_nmf_22-04-2025"

@run_and_track(
    algorithm_names=algs, 
    path_store="Results/",
    name_store=name,
    skip=skip,
    **variables
)
def one_run(rank = 6,
            NbIter = 200,
            NbIter_HALS = 50,
            NbIter_inner = 10,
            delta = 0,
            epsilon = 1e-16,
            tol=0,
            seed=1
            ):
    # Print
    verbose=True
    print_it=10
    # Seeding
    rng = np.random.RandomState(seed+20)
    # Init
    Wini = Wref + 0.1*np.random.rand(m,rank)
    Hini = rng.rand(rank, n)
    lamb = opt_scaling_fro(M, Wref@Hini)
    Hini = lamb*Hini

    # Frobenius algorithms
    error0, W0, H0, toc0, cnt0 = nmf_f.NMF_Lee_Seung(M,  Wini, Hini, NbIter, NbIter_inner,tol=tol, legacy=False, epsilon=epsilon, verbose=verbose, delta=delta, print_it=print_it)   
    error1, W1, H1, toc1, cnt1 = nmf_f.NMF_proposed_Frobenius(M, Wini, Hini, NbIter, NbIter_inner, tol=tol, delta=delta, verbose=verbose, print_it=print_it, gamma=1.9)
    error2, W2, H2, toc2, cnt2 = nmf_f.NMF_proposed_Frobenius(M, Wini, Hini, NbIter, NbIter_inner, tol=tol, method="AMUSOM", delta=delta, verbose=verbose, print_it=print_it, gamma=1.9)
    #error3, W3, H3, toc3, cnt3  = nmf_f.NeNMF_optimMajo(M, Wini, Hini, tol=tol, itermax=NbIter, nb_inner=NbIter_inner, epsilon=epsilon, verbose=verbose, delta=delta, print_it=print_it, gamma=1)
    error4, W4, H4, toc4, cnt4  = nmf_f.Grad_descent(M , Wini, Hini, NbIter, NbIter_inner, tol=tol, epsilon=epsilon, verbose=verbose, delta=delta, print_it=print_it)
    error5, W5, H5, toc5, cnt5  = nmf_f.NeNMF(M, Wini, Hini, tol=tol, nb_inner=NbIter_inner, itermax=NbIter, epsilon=epsilon, verbose=verbose, delta=delta, print_it=print_it)
    W6, H6, error6, toc6, cnt6 = nn_fac.nmf.nmf(M, rank, init="custom", U_0=np.copy(Wini), V_0=np.copy(Hini), n_iter_max=NbIter, tol=tol, update_rule='hals', beta=2, return_costs=True, NbIter_inner=NbIter_inner, verbose=verbose, delta=delta, print_it=print_it)

    # KL algorithms
    #error7, W7, H7, toc7, cnt7 = nmf_kl.Lee_Seung_KL(M, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=print_it)
    #error8, W8, H8, toc8, cnt8 = nmf_kl.Proposed_KL(M, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=print_it, gamma=1.9)
    #error9, W9, H9, toc9, cnt9 = nmf_kl.Proposed_KL(M, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=print_it, gamma=1.9, method="trueMU")

    return {
        "errors": [error0, error1, error2, error4, error5, error6],
        "timings": [toc0, toc1, toc2, toc4, toc5, toc6],
    }

df = pd.read_pickle("Results/"+name)

# interpolation
ovars_iterp = ["algorithm"]
df = interpolate_time_and_error(df, npoints = 100, adaptive_grid=True, groups=ovars_iterp)

# Making a convergence plot dataframe
df_l2_conv = df_to_convergence_df(df, groups=True, groups_names=[], other_names=[], err_name="errors_interp", time_name="timings_interp")
df_l2_conv = df_l2_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})

df_l2_conv_it = pp.df_to_convergence_df(df, groups=True, groups_names=[], other_names=[])

df_l2_conv_median_time = median_convergence_plot(df_l2_conv, type_x="timings")
df_l2_conv_median_it = pp.median_convergence_plot(df_l2_conv_it, type_x="iterations")

# dirty hack to get the naming of algorithms in the same order as the NLS problem
# plotly takes the order of the algorithms as they appear in the dataframe, so we reorder them
# we swap row 5 with row 3, 3 with 4 and 4 with 5
a = df_l2_conv_median_time
b, c, d = a.iloc[3], a.iloc[4], a.iloc[5]

temp1 = b.copy()
temp2 = c.copy()
a.iloc[3] = d
a.iloc[4] = b
a.iloc[5] = c


# ----------------------- Plot --------------------------- #
# Convergence plots with all runs
pxfig = px.line(df_l2_conv_median_time,
            #line_group="groups",
            x="timings",
            y="errors",
            color='algorithm',
            line_dash='algorithm',
            log_y=True,
            log_x=True,
            )

pxfigit = px.line(df_l2_conv_it,  #_median_it, 
            x="it", 
            y= "errors", 
            color='algorithm',
            line_dash='algorithm',
            log_y=True,
            line_group="groups"
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
    title_text = "NMF",
    font_size = 10,
    width=450/2, # in px
    height=350,
    #xaxis=dict(range=[0, 40], title_text="Time (s)"),
    xaxis=dict(title_text="Time (s)"),
    yaxis=dict(range=np.log10([0.00145,0.0030]), title_text="n. Loss")
    #yaxis=dict(title_text="Fit")
)

pxfig.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfig.update_yaxes(
    matches=None,
    showticklabels=True
)

pxfigit.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfigit.update_layout(
    title_text = "NLS",
    font_size = 12,
    width=450*1.62/2, # in px
    height=450,
    #xaxis=dict(range=[0,1.0], title_text="Time (s)"),
    #yaxis=dict(title_text="Fit")
)

pxfigit.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfigit.update_yaxes(
    matches=None,
    showticklabels=True
)
pxfig.write_image("Results/"+name+"_fro.pdf")
pxfig.write_image("Results/"+name+"_fro.pdf")
pxfigit.write_image("Results/"+name+"_fro_it.pdf")
pxfig.show()
pxfigit.show()