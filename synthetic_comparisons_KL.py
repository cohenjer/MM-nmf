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
                m = [50,100],
                n = [50,100],
                r = [5,10],
                sigma = [0,1e-2,1],
                NbIter_inner = [2,10]
                )
def one_run(m=100,n=100,r=10,sigma=0, NbIter=3000,tol=0,NbIter_inner=10, verbose=True, show_it=1000):
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

# Adding in results errors at specific timings and iterations
#df = error_at_time_or_it(df, time_stamps=[0.1, 0.5, 1], it_stamps=[10, 50, 300])

# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
ovars = ["NbIter_inner","r","sigma"]
df_conv = df_to_convergence_df(df, max_time=0.9, 
                                filters={"n":50, "m":100},
                                groups=True, groups_names=ovars, other_names=ovars)

# ----------------------- Plot --------------------------- #
# TODO: go to plotly
fig_winner = plot_speed_comparison(thresh, scores_time, scores_it, legend=algs)
fig_winner.show()

## Boxplots with errors after fixed time/iterations
# Not so interesting
# TODO: automate these plots
#xax = "sigma"

#fig_box = px.box(df, y="err_at_time_0.1", color="algorithm", x=xax, log_y=True, template="plotly_white")
#fig_box2 = px.box(df, y="err_at_time_0.5", color="algorithm", x=xax,log_y=True, template="plotly_white")
#fig_box3 = px.box(df, y="err_at_time_1", color="algorithm", x=xax, log_y=True, template="plotly_white")
#fig_box.update_xaxes(type='category')
#fig_box2.update_xaxes(type='category')
#fig_box3.update_xaxes(type='category')
#fig_box.show()
#fig_box2.show()
#fig_box3.show()
#fig_box_it = px.box(df, y="err_at_it_10", color="algorithm", x=xax,log_y=True, template="plotly_white")
#fig_box_it_2 = px.box(df, y="err_at_it_50", color="algorithm", x=xax, log_y=True, template="plotly_white")
#fig_box_it_3 = px.box(df, y="err_at_it_300", color="algorithm", x=xax, log_y=True, template="plotly_white")
#fig_box_it.update_xaxes(type='category')
#fig_box_it_2.update_xaxes(type='category')
#fig_box_it_3.update_xaxes(type='category')
#fig_box_it.show()
#fig_box_it_2.show()
#fig_box_it_3.show()

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