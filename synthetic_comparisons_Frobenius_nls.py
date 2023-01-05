import numpy as np
from matplotlib import pyplot as plt
import NLS_Frobenius as nls_f 
import nn_fac
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Personnal comparison toolbox
# you can get it at 
# https://github.com/cohenjer/shootout
from shootout.methods.runners import run_and_track

plt.close('all')
# --------------------- Choose parameters for grid tests ------------ #
algs = ["Lee_Sung","fastMU_ex","GD", "NeNMF", "HALS", "fastMU", "fastMU_min"]
nb_seeds = 10  # Change this to >0 to run experiments
name = "l2_nls_run-01-05-2023"
@run_and_track(algorithm_names=algs, path_store="Results/", name_store=name,
                nb_seeds=nb_seeds, seeded_fun=True,
                mnr = [[200,100,5], [1000,400,20]],
                SNR = [100, 30],
                NbIter = [200],
                delta = 0
                )
def one_run(mnr=[100,100,10],SNR=50, NbIter=3000,tol=0,NbIter_inner=10, seed=1,delta=0.4):
    m, n, r = mnr
    # Fixed the signal 
    rng = np.random.RandomState(seed+20)
    Worig = rng.rand(m, r) 
    Horig = rng.rand(r, n)  
    Vorig = Worig.dot(Horig)

    # prints
    verbose = True

    # Initialization for H0 as a random matrix
    Hini = rng.rand(r, n)
    
    # adding Gaussian noise to the observed data
    N = rng.randn(m,n)
    sigma = 10**(-SNR/20)*np.linalg.norm(Vorig)/np.linalg.norm(N)
    V = Vorig + sigma*N

    # One noise, one init; NMF is not unique and nncvx so we will find several results
    error0, H0, toc0 = nls_f.NMF_Lee_Seung(V,  Worig, Hini, NbIter, legacy=False, delta=delta, verbose=verbose)
    error1, H1, toc1  = nls_f.NeNMF_optimMajo(V, Worig, Hini, itermax=NbIter, delta=delta, verbose=verbose) #deactivate max_step
    error2, H2, toc2  = nls_f.Grad_descent(V , Worig, Hini, NbIter, delta=delta, verbose=verbose)
    error3, H3, toc3  = nls_f.NeNMF(V, Worig, Hini, itermax=NbIter, delta=delta, verbose=verbose)
    H4, _, _, _, error4, toc4 = nn_fac.nnls.hals_nnls_acc(Worig.T@V, Worig.T@Worig, np.copy(Hini), maxiter=NbIter, return_error=True, delta=delta, M=V)
    error5, H5, toc5 = nls_f.NMF_proposed_Frobenius(V, Worig, Hini, NbIter, use_LeeS=False, delta=delta, verbose=verbose)
    error6, H6, toc6 = nls_f.NMF_proposed_Frobenius(V, Worig, Hini, NbIter, use_LeeS=True, delta=delta, verbose=verbose, gamma=1.2) # use gamma=1?

    return {"errors" : [error0, error1, error2, error3, error4, error5, error6], 
            "timings" : [toc0, toc1, toc2, toc3, toc4, toc5, toc6],
            }


# -------------------- Post-Processing ------------------- #
import pandas as pd
import shootout.methods.post_processors as pp

df = pd.read_pickle("Results/"+name)
nb_seeds = df["seed"].max()+1 # get nbseed from data

# TODO: shootout plots
## Using shootout for plotting and postprocessing
#thresh = np.logspace(-3,-8,50) 
#scores_time, scores_it, timings, iterations = pp.find_best_at_all_thresh(df,thresh, nb_seeds)

# ----------------------- Plot --------------------------- #
# TODO: go to plotly
#fig_winner = plot_speed_comparison(thresh, scores_time, scores_it, legend=algs)
#fig_winner.show()

# Adding in results errors at specific timings and iterations
#df = pp.error_at_time_or_it(df, time_stamps=[0.1, 0.5, 1], it_stamps=[10, 50, 300])

# Group up columns
df = pp.regroup_columns(df, keys=["mnr"], how_many=3)

# Interpolating time (choose fewer points for better vis), adaptive grid since time varies across plots
df = pp.interpolate_time_and_error(df, npoints = 100, adaptive_grid=True)

# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
ovars = ["mnr", "SNR"]
df_conv = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars, err_name="errors_interp", time_name="timings_interp")
df_conv = df_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})

# Median plot
df_conv_median_time = pp.median_convergence_plot(df_conv, type="timings", mean=True)

# Convergence plots with all runs
pxfig = px.line(df_conv_median_time, 
            x="timings", 
            y= "errors", 
            color='algorithm',
            facet_col="mnr",
            facet_row="SNR",
            log_y=True,
            #log_x=True,
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