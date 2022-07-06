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
algs = ["Lee_Sung", "Fevotte_Idier", "Proposed"]
nb_seeds = 3
@run_and_track(algorithm_names=algs, path_store="Results/", name_store="KL_run_unif_noise",
                nb_seeds=0,#nb_seeds, # Change this to >0 to run experiments
                m = [50],
                n = [50],
                r = [5],
                #sigma = [0,1e-2,1], # for poisson
                sigma = [0,1e-6,1e-3], # for uniform
                NbIter_inner = [1,10,20]
                )
def one_run(m=100,n=100,r=10,sigma=0, NbIter=3000,tol=0,NbIter_inner=10, verbose=False, show_it=1000):
    # Fixed the signal 
    # here it is dense, TODO: try sparse
    Worig = np.random.rand(m, r) 
    Horig = np.random.rand(r, n)  
    Vorig = Worig.dot(Horig)

    # Initialization for H0 as a random matrix
    Hini = np.random.rand(r, n)
    Wini = np.random.rand(m, r) #sparse.random(rV, cW, density=0.25).toarray() 
    
    # adding Poisson noise to the observed data
    # TODO: discuss noise choice
    #N = np.random.poisson(sigma,size=Vorig.shape) # integers, should we scale?
    N = sigma*np.random.rand(m,n) # uniform, should we scale?
    V = Vorig + N

    # TODO: compute SNR/use SNR to set up noise

    # One noise, one init; NMF is not unique and nncvx so we will find several results
    error0, W0, H0, toc0 = nmf_kl.Lee_Seung_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it)
    error1, W1, H1, toc1 = nmf_kl.Fevotte_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it)
    #error2, W2, H2, toc2 = nmf_kl.NeNMF_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol)
    error3, W3, H3, toc3 = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it)

    return {"errors" : [error0, error1, error3], 
            "timings" : [toc0, toc1, toc3],
            #"noise": [N,N,N]
            }


# -------------------- Post-Processing ------------------- #
import pandas as pd

name = "KL_run_unif_noise"
df = pd.read_pickle("Results/"+name)

# Using shootout for plotting and postprocessing
thresh = np.logspace(-3,-8,50) 
scores_time, scores_it, timings, iterations = find_best_at_all_thresh(df,thresh, nb_seeds)

# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
ovars = ["NbIter_inner","r","sigma"]
df_conv = df_to_convergence_df(df, max_time=0.9, 
                                filters={"n":50, "m":50},
                                groups=True, groups_names=ovars, other_names=ovars)

# ----------------------- Plot --------------------------- #
# TODO: go to plotly
fig_winner = plot_speed_comparison(thresh, scores_time, scores_it, legend=algs)
fig_winner.show()

# Convergence plots with all runs
pxfig = px.line(df_conv, line_group="groups", x="timings", y= "errors", color='algorithm',
            facet_col="NbIter_inner", facet_row="sigma",
            log_y=True,
            height=1000)
pxfig.update_layout(font = dict(size = 20))
pxfig2 = px.line(df_conv, line_group="groups", x="it", y= "errors", color='algorithm',
            facet_col="NbIter_inner", facet_row="sigma",
            log_y=True,
            height=1000)
pxfig2.update_layout(font = dict(size = 20))
pxfig.show()
pxfig2.show()

plt.show()