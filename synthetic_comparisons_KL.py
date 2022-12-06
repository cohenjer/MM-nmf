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
import shootout.methods.post_processors as pp
from shootout.methods.plotters import plot_speed_comparison

plt.close('all')

# --------------------- Choose parameters for grid tests ------------ #
algs = ["Lee_Sung", "Proposed_with_max", "Proposed_no_max"]
nb_seeds = 10  # Change this to >0 to run experiments

name = "KL_run_06-12-2022"
@run_and_track(algorithm_names=algs, path_store="Results/", name_store=name,
                add_track = {"distribution" : "uniform"},
                nb_seeds=nb_seeds,
                mnr = [[200,100,5]],#,[1000,400,20]],
                NbIter_inner = [100], # for Lee and Seung also
                SNR = [100],
                delta = 0.1,
                seeded_fun=True,
                )
def one_run(mnr=[100,100,5],SNR=50, NbIter=3000, tol=0, NbIter_inner=10, verbose=True, show_it=100, delta=0, seed=1):
    m, n, r = mnr
    # Fixed the signal 
    rng = np.random.RandomState(seed+20)
    Worig = rng.rand(m, r) 
    Horig = rng.rand(r, n)  
    Vorig = Worig.dot(Horig)

    # Initialization for H0 as a random matrix
    Hini = rng.rand(r, n)
    Wini = rng.rand(m, r) #sparse.random(rV, cW, density=0.25).toarray() 
    
    # adding Poisson noise to the observed data
    #N = np.random.poisson(1,size=Vorig.shape) # integers
    N = rng.rand(m,n) # uniform
    sigma = 10**(-SNR/20)*np.linalg.norm(Vorig)/np.linalg.norm(N)
    V = Vorig + sigma*N

    # One noise, one init; NMF is not unique and nncvx so we will find several results
    error0, W0, H0, toc0, cnt0 = nmf_kl.Lee_Seung_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it, delta=delta)
    error1, W1, H1, toc1, cnt1 = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it, delta=delta, use_LeeS=True)
    error2, W2, H2, toc2, cnt2 = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it, delta=delta, use_LeeS=False)

    return {"errors" : [error0, error1, error2], 
            "timings" : [toc0, toc1, toc2],
            "cnt" :  [cnt0[::10], cnt1[::10], cnt2[::10]]
           }


# -------------------- Post-Processing ------------------- #
import pandas as pd

df = pd.read_pickle("Results/"+name)
nb_seeds = df["seed"].max()+1 # get nbseed from data

# Grouping columns
df = pp.regroup_columns(df, keys=["mnr"], how_many=3)

# Making a convergence plot dataframe
df = pp.interpolate_time_and_error(df, npoints = 100, adaptive_grid=True)

# We will show convergence plots for various sigma values, with only n=100
ovars = ["mnr"]
df_conv = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars,err_name="errors_interp", time_name="timings_interp")
df_conv = df_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})

# Median plot
df_conv_median_time = pp.median_convergence_plot(df_conv, type="timings")

# Convergence plots with all runs
pxfig = px.line(df_conv_median_time, 
            x="timings", 
            y= "errors", 
            color='algorithm',
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
    error_y_thickness = 0.3,
)
pxfig.update_xaxes(matches=None)
pxfig.update_yaxes(matches=None)
pxfig.update_xaxes(showticklabels=True)
pxfig.update_yaxes(showticklabels=True)
pxfig.show()