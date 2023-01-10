import numpy as np
from matplotlib import pyplot as plt
import NMF_KL as nmf_kl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import plotly.io as pio
pio.kaleido.scope.mathjax = None

# Personnal comparison toolbox
# you can get it at 
# https://github.com/cohenjer/shootout
from shootout.methods.runners import run_and_track
from shootout.methods.post_processors import find_best_at_all_thresh, df_to_convergence_df, error_at_time_or_it
from shootout.methods.plotters import plot_speed_comparison
from shootout.methods.post_processors import interpolate_time_and_error, median_convergence_plot

plt.close('all')

# --------------------- Choose parameters for grid tests ------------ #
algs = ["Proposed-testing delta and inner iters"]
if len(sys.argv)==1:
    nb_seeds = 0 #no run
else:
    nb_seeds = int(sys.argv[1])  # Change this to >0 to run experiments

name = "KL_run_delta-choice-01-06-2023"
variables = {
    'mnr' : [[200,100,5]],
    'NbIter_inner' : [100],
    'delta' : [0,0.001,0.01,0.05,0.1,0.3,0.6,0.9],
    'SNR' : [100],
}

@run_and_track(algorithm_names=algs, path_store="Results/", name_store=name,
                add_track = {"distribution" : "uniform"},
                nb_seeds=nb_seeds,
                seeded_fun=True,
                single_method=True,
                **variables
                )
def one_run(mnr=[100,100,5],SNR=50, NbIter=20000, tol=0, NbIter_inner=10, verbose=False, show_it=1000, delta=0.4, seed=1):
    NbIter = int(NbIter*(delta+0.1)) # will be 1500 for 100 iter inner
    m, n, r = mnr
    # Fixed the signal 
    rng = np.random.RandomState(seed+20)
    Worig = rng.rand(m, r) 
    Horig = rng.rand(r, n)  
    Vorig = Worig.dot(Horig)

    # Initialization for H0 as a random matrix
    Hini = rng.rand(r, n)
    Wini = rng.rand(m, r) #sparse.random(rV, cW, density=0.25).toarray() 
    
    # adding noise to the observed data
    #N = np.random.poisson(1,size=Vorig.shape) # integers
    N = rng.rand(m,n) # uniform
    sigma = 10**(-SNR/20)*np.linalg.norm(Vorig)/np.linalg.norm(N)
    V = Vorig + sigma*N

    # One noise, one init; NMF is not unique and nncvx so we will find several results
    error, _, _, toc, cnt = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it, delta=delta, gamma=1.9)

    return {"errors" : error, 
            "timings" : toc,
            "cnt": cnt[::10], # subsample otherwise too big
            }


# -------------------- Post-Processing ------------------- #
import pandas as pd
import plotly.io as pio
pio.templates.default= "plotly_white"

# Load results
df = pd.read_pickle("Results/"+name)
nb_seeds = df["seed"].max()+1 # get nbseed from data

# TODO improve: list variables are split in runs, and therefore dont match between df and variables
variables.pop('mnr')

# Interpolating time (choose fewer points for better vis), adaptive grid since time varies across plots
df = interpolate_time_and_error(df, npoints = 100, adaptive_grid=True)

# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
ovars = list(variables.keys())
df_conv = df_to_convergence_df(
    df, 
    max_time=np.Inf,
    groups=True, groups_names=ovars, other_names=ovars,
    err_name="errors_interp", time_name="timings_interp"
    )
df_conv = df_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})

# Converting to median for iterations and timings
df_conv_median_time = median_convergence_plot(df_conv, type="timings")

# Convergence plots with all runs
pxfig = px.line(df_conv_median_time, 
            x="timings", 
            y= "errors", 
            color='delta',
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
    font_size = 12,
    width=450*1.62/2, # in px
    height=450,
    xaxis=dict(range=[0,5], title_text="Time (s)"),
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

pxfig.write_image("Results/"+name+".pdf")
pxfig.write_image("Results/"+name+".pdf")
pxfig.show()

# Figure showing cnt for each algorithm
# 1. make long format for cnt
# TODO: improve shootout to better handle this case
df_conv_cnt = df_to_convergence_df(df, groups=True, groups_names=ovars, err_name="cnt", other_names=ovars, time_name=False, max_time=False)
# 2. median plots
df_conv_median_cnt = median_convergence_plot(df_conv_cnt, type=False, err_name="cnt")

pxfig2 = px.line(df_conv_median_cnt, 
            x="it", 
            y= "cnt", 
            color='delta',
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
    font_size = 12,
    width=450*1.62/2, # in px
    height=450,
    xaxis=dict(range=[0,2000], title_text="Outer iteration"),
    yaxis=dict(title_text="Number of inner loops")
)

pxfig2.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfig2.update_yaxes(
    matches=None,
    showticklabels=True
)

pxfig2.write_image("Results/"+name+"_cnt.pdf")
pxfig2.show()
