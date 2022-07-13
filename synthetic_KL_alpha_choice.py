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
algs = ["alpha_1", "alpha_data", "alpha_factors", "alpha_1.8", "alpha_2", "alpha_2.5"]
nb_seeds = 10
name = "KL_run_alpha_comparisons"
@run_and_track(algorithm_names=algs, path_store="Results/", name_store=name,
                nb_seeds=nb_seeds, # Change this to >0 to run experiments
                m = [100],
                n = [50],
                r = [8],
                #sigma = [0,1e-2,1], # for poisson
                sigma = [1e-3],#,1e-3], # for uniform
                NbIter_inner = [10],
                delta=[0],
                )
def one_run(m=100,n=100,r=10,sigma=0, NbIter=3000,tol=0,NbIter_inner=10, verbose=True, show_it=1000, delta=0.4):
    # Fixed the signal 
    Worig = np.random.rand(m, r) 
    Horig = np.random.rand(r, n)  
    Vorig = Worig.dot(Horig)

    # Initialization for H0 as a random matrix
    Hini = np.random.rand(r, n)
    Wini = np.random.rand(m, r) #sparse.random(rV, cW, density=0.25).toarray() 
    
    # adding Poisson noise to the observed data
    #N = np.random.poisson(sigma,size=Vorig.shape) # integers, should we scale?
    N = sigma*np.random.rand(m,n) # uniform, should we scale?
    V = Vorig + N
    # TODO: compute SNR/use SNR to set up noise

    # One noise, one init; NMF is not unique and nncvx so we will find several results
    error1, _, _, toc1, _ = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it, delta=delta, alpha_strategy=1)
    error2, _, _, toc2, _ = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it, delta=delta, alpha_strategy="data_sum")
    error3, _, _, toc3, _ = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it, delta=delta, alpha_strategy="factors_sum")
    error4, _, _, toc4, _ = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it, delta=delta, alpha_strategy=1.8)
    error5, _, _, toc5, _ = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it, delta=delta, alpha_strategy=2)
    error6, _, _, toc6, _ = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it, delta=delta, alpha_strategy=2.5)

    return {"errors" : [error1, error2, error3, error4, error5, error6], 
            "timings" : [toc1, toc2, toc3, toc4, toc5, toc6],
            }


# -------------------- Post-Processing ------------------- #
import pandas as pd

df = pd.read_pickle("Results/"+name)
nb_seeds = df["seed_idx"].max()+1 # get nbseed from data

# Using shootout for plotting and postprocessing
thresh = np.logspace(-1,-4,50) 
scores_time, scores_it, timings, iterations = find_best_at_all_thresh(df,thresh, nb_seeds)

# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
ovars = ["r"] #anything here, should fix shootout
df_conv = df_to_convergence_df(df, max_time=np.Inf,  
                                groups=True, groups_names=ovars, other_names=ovars)

# ----------------------- Plot --------------------------- #
# TODO: go to plotly
fig_winner = plot_speed_comparison(thresh, scores_time, scores_it, legend=algs)
fig_winner.show()

# Convergence plots with all runs
pxfig = px.line(df_conv, line_group="groups", x="timings", y= "errors", color='algorithm',
            facet_col="seed_idx", facet_row=None,
            log_y=True,
            height=1000)
pxfig.update_layout(font = dict(size = 20))
pxfig2 = px.line(df_conv, line_group="groups", x="it", y= "errors", color='algorithm',
            facet_col="seed_idx", facet_row=None,
            log_y=True,
            height=1000)
pxfig2.update_layout(font = dict(size = 20))
pxfig.show()
pxfig2.show()

plt.show()