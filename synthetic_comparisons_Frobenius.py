import numpy as np
from matplotlib import pyplot as plt
import NMF_Frobenius as nmf_f 
import nn_fac
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
@run_and_track(algorithm_names=["Lee_Sung","Proposed","GD", "NeNMF", "HALS"],path_store="Results/", name_store="Euclidean_run_test",
                nb_seeds=0, # Change this to >0 to run experiments
                m = [100],
                n = [100,200],
                r = [10],
                sigma = [0,1e-7,1e-4],
                NbIter_inner = 10
                )
def one_run(m=100,n=100,r=10,sigma=0, NbIter=3000,NbIter_hals=1000,tol=0,NbIter_inner=10):
    # Fixed the signal 
    Worig = np.random.rand(m, r) 
    Horig = np.random.rand(r, n)  
    Vorig = Worig.dot(Horig)

    # Initialization for H0 as a random matrix
    Hini = np.random.rand(r, n)
    Wini = np.random.rand(m, r) #sparse.random(rV, cW, density=0.25).toarray() 
    
    # adding Gaussian noise to the observed data
    N = sigma*np.random.randn(m,n)
    V = Vorig + N

    # One noise, one init; NMF is not unique and nncvx so we will find several results
    error0, W0, H0, toc0 = nmf_f.NMF_Lee_Seung(V,  Wini, Hini, NbIter, NbIter_inner,tol=tol, legacy=False)
    error1, W1, H1, toc1  = nmf_f.NeNMF_optimMajo(V, Wini, Hini, tol=tol, itermax=NbIter, nb_inner=NbIter_inner)
    #error1, W1, H1, toc1 = nmf_f.NMF_proposed_Frobenius(V, Wini, Hini, NbIter, NbIter_inner, tol=tol)
    error2, W2, H2, toc2  = nmf_f.Grad_descent(V , Wini, Hini, NbIter, NbIter_inner, tol=tol)
    error3, W3, H3, toc3  = nmf_f.NeNMF(V, Wini, Hini, tol=tol, nb_inner=NbIter_inner, itermax=NbIter)
    # Fewer max iter cause too slow
    W4, H4, error4, toc4 = nn_fac.nmf.nmf(V, r, init="custom", U_0=np.copy(Wini), V_0=np.copy(Hini), n_iter_max=NbIter_hals, tol=tol, update_rule='hals',beta=2, return_costs=True, NbIter_inner=NbIter_inner)

    return {"errors" : [error0, error1, error2, error3, error4], 
            "timings" : [toc0,toc1,toc2,toc3,toc4]}


# -------------------- Post-Processing ------------------- #
import pandas as pd

name = "run-2022-07-05_12-08"
df = pd.read_pickle("Results/"+name)

# Using shootout for plotting and postprocessing
thresh = np.logspace(-3,-8,50) 
scores_time, scores_it, timings, iterations = find_best_at_all_thresh(df,thresh, 5)

# Adding in results errors at specific timings and iterations
df = error_at_time_or_it(df, time_stamps=[0.1, 0.5, 1], it_stamps=[10, 50, 300])

# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
ovars = ["sigma"]
df_conv = df_to_convergence_df(df, max_time=0.9, filters={"n":100}, groups=True, groups_names=ovars, other_names=ovars)

# ----------------------- Plot --------------------------- #
# TODO: go to plotly
fig_winner = plot_speed_comparison(thresh, scores_time, scores_it, legend=["Lee_Sung","Proposed","GD", "NeNMF", "HALS"])
fig_winner.show()

## Boxplots with errors after fixed time/iterations
# TODO: automate these plots
xax = "sigma"

fig_box = px.box(df, y="err_at_time_0.1", color="algorithm", x=xax, log_y=True, template="plotly_white")
fig_box2 = px.box(df, y="err_at_time_0.5", color="algorithm", x=xax,log_y=True, template="plotly_white")
fig_box3 = px.box(df, y="err_at_time_1", color="algorithm", x=xax, log_y=True, template="plotly_white")
fig_box.update_xaxes(type='category')
fig_box2.update_xaxes(type='category')
fig_box3.update_xaxes(type='category')
fig_box.show()
fig_box2.show()
fig_box3.show()
fig_box_it = px.box(df, y="err_at_it_10", color="algorithm", x=xax,log_y=True, template="plotly_white")
fig_box_it_2 = px.box(df, y="err_at_it_50", color="algorithm", x=xax, log_y=True, template="plotly_white")
fig_box_it_3 = px.box(df, y="err_at_it_300", color="algorithm", x=xax, log_y=True, template="plotly_white")
fig_box_it.update_xaxes(type='category')
fig_box_it_2.update_xaxes(type='category')
fig_box_it_3.update_xaxes(type='category')
fig_box_it.show()
fig_box_it_2.show()
fig_box_it_3.show()

# Convergence plots with all runs
pxfig = px.line(df_conv, line_group="groups", x="timings", y= "errors", color='algorithm',facet_col="sigma",
              log_y=True,
              height=1000)
pxfig.update_layout(font = dict(size = 20))
pxfig2 = px.line(df_conv, line_group="groups", x="it", y= "errors", color='algorithm',facet_col="sigma",
              log_y=True,
              height=1000)
pxfig2.update_layout(font = dict(size = 20))
pxfig.show()
pxfig2.show()

plt.show()