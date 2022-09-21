import numpy as np
from matplotlib import pyplot as plt
import NMF_KL as nmf_kl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Personnal comparison toolbox
# you can get it at 
# https://github.com/cohenjer/shootout
from shootout.methods.runners import run_and_track
from shootout.methods.post_processors import find_best_at_all_thresh, df_to_convergence_df, error_at_time_or_it
from shootout.methods.plotters import plot_speed_comparison

plt.close('all')

# --------------------- Choose parameters for grid tests ------------ #
algs = ["NMF Pham"]
nb_seeds = 0
name = "KL_run_alpha_comparisons_19-09-2022"
@run_and_track(algorithm_names=algs, path_store="Results/", name_store=name,
                add_track = {"distribution" : "uniform"},
                nb_seeds=nb_seeds, # Change this to >0 to run experiments
                mnr = [[10,10,2],[100,50,5],[200,150,10]],
                SNR = [100],
                NbIter_inner = [100],
                delta = [0.1],
                alpha = [1e-2,1e-1,1,1.9,2,3,"data_sum","factors_sum"],
                seeded_fun = True
                )
def one_run(mnr=[100,100,10],SNR=100, NbIter=2000,tol=0,NbIter_inner=10, verbose=True,
            show_it=1000, delta=0.4, alpha=1, seed=1):
    # Fixed the signal 
    m, n, r = mnr
    rng = np.random.RandomState(seed+20)
    Worig = rng.rand(m, r) 
    Horig = rng.rand(r, n)  
    Vorig = Worig.dot(Horig)

    # Initialization for H0 as a random matrix
    Hini = rng.rand(r, n)
    Wini = rng.rand(m, r) #sparse.random(rV, cW, density=0.25).toarray() 
    
    # adding Poisson noise to the observed data
    #N = np.random.poisson(sigma,size=Vorig.shape) # integers, should we scale?
    N = rng.rand(m,n) # uniform, should we scale?
    sigma = 10**(-SNR/20)*np.linalg.norm(Vorig)/np.linalg.norm(N)
    V = Vorig + sigma*N

    # One noise, one init; NMF is not unique and nncvx so we will find several results
    error1, _, _, toc1, cnt1 = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it, delta=delta, alpha_strategy=alpha)

    return {"errors" : error1, "timings" : toc1, "cnt" : cnt1}


# -------------------- Post-Processing ------------------- #
import pandas as pd
from shootout.methods import post_processors as pp

df = pd.read_pickle("Results/"+name)
nb_seeds = df["seed"].max()+1 # get nbseed from data



# Making a convergence plot dataframe
# First we interpolate
df = pp.interpolate_time_and_error(df, npoints = 100, adaptive_grid=True)

# We will show convergence plots for various sigma values, with only n=100
ovars = ["alpha", "mnr_0", "mnr_1", "mnr_2"] #anything here, should fix shootout
df_conv = df_to_convergence_df(
    df, max_time=np.Inf, err_name="errors_interp", time_name="timings_interp",
    groups=True, groups_names=ovars, other_names=ovars
    )
df_conv = df_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})

# Converting to median for iterations and timings
df_conv_median_time = pp.median_convergence_plot(df_conv, type="timings")

# regroup split dimensions (at the end because lists are nightmare-ish in dataframes)
df_conv_median_time = pp.regroup_columns(df_conv_median_time, keys=["mnr"], how_many=3)

# Convergence plots with all runs
pxfig = px.line(df_conv_median_time, 
            x="timings", 
            y= "errors", 
            color='alpha',
            facet_row="mnr",
            log_y=True,
            error_y="q_errors_p", 
            error_y_minus="q_errors_m", 
            template="plotly_white",
            height=1000)
pxfig.update_layout(
    font_size = 20,
    width=1200, # in px
    height=900,
    )
# smaller linewidth
pxfig.update_traces(
    selector=dict(),
    line_width=3,
    error_y_thickness = 0.1,
)
pxfig.update_xaxes(matches=None)
pxfig.update_yaxes(matches=None)
pxfig.update_xaxes(showticklabels=True)
pxfig.update_yaxes(showticklabels=True)
pxfig.show()