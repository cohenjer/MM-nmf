import numpy as np
from matplotlib import pyplot as plt
import NMF_Frobenius as nmf_f 
import nn_fac
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
from shootout.methods.plotters import plot_speed_comparison

plt.close('all')
# --------------------- Choose parameters for grid tests ------------ #
algs = ["MU_Fro","fastMU_Fro_ex","GD_Fro", "NeNMF_Fro", "HALS", "fastMU_Fro", "fastMU_Fro_min"]
if len(sys.argv)==1:
    nb_seeds = 0 #no run
else:
    nb_seeds = int(sys.argv[1])  # Change this to >0 to run experiments
name = "l2_run-01-06-2023"
@run_and_track(algorithm_names=algs, path_store="Results/", name_store=name,
                nb_seeds=nb_seeds, seeded_fun=True,
                mnr = [[200,100,5], [1000,400,20]],
                SNR = [100],
                NbIter_inner = [100],
                delta = 0.1
                )
def one_run(mnr=[100,100,10],SNR=50, NbIter=3000,tol=0,NbIter_inner=10, seed=1,delta=0.4):
    m, n, r = mnr
    # Fixed the signal 
    rng = np.random.RandomState(seed+20)
    Worig = rng.rand(m, r) 
    Horig = rng.rand(r, n)  
    Vorig = Worig.dot(Horig)

    # prints
    verbose = False

    # Initialization for H0 as a random matrix
    Hini = rng.rand(r, n)
    Wini = rng.rand(m, r) #sparse.random(rV, cW, density=0.25).toarray() 
    
    # adding Gaussian noise to the observed data
    N = rng.randn(m,n)
    sigma = 10**(-SNR/20)*np.linalg.norm(Vorig)/np.linalg.norm(N)
    V = Vorig + sigma*N

    # One noise, one init; NMF is not unique and nncvx so we will find several results
    error0, W0, H0, toc0, cnt0 = nmf_f.NMF_Lee_Seung(V,  Wini, Hini, NbIter, NbIter_inner,tol=tol, legacy=False, delta=delta, verbose=verbose)
    error2, W2, H2, toc2, cnt2  = nmf_f.Grad_descent(V , Wini, Hini, NbIter, NbIter_inner, tol=tol, delta=delta, verbose=verbose)
    error3, W3, H3, toc3, cnt3  = nmf_f.NeNMF(V, Wini, Hini, tol=tol, nb_inner=NbIter_inner, itermax=NbIter, delta=delta, verbose=verbose)
    # Fewer max iter because too slow
    W4, H4, error4, toc4, cnt4 = nn_fac.nmf.nmf(V, r, init="custom", U_0=np.copy(Wini), V_0=np.copy(Hini), n_iter_max=NbIter, tol=tol, update_rule='hals',beta=2, return_costs=True, NbIter_inner=NbIter_inner, delta=delta, verbose=verbose)

    # Proposed
    error1, W1, H1, toc1, cnt1  = nmf_f.NeNMF_optimMajo(V, Wini, Hini, tol=tol, itermax=NbIter, nb_inner=NbIter_inner, delta=delta, verbose=verbose, use_best=False, gamma=1)
    error5, W5, H5, toc5, cnt5 = nmf_f.NMF_proposed_Frobenius(V, Wini, Hini, NbIter, NbIter_inner, tol=tol, use_LeeS=False, delta=delta, verbose=verbose, gamma=1.9)
    error6, W6, H6, toc6, cnt6 = nmf_f.NMF_proposed_Frobenius(V, Wini, Hini, NbIter, NbIter_inner, tol=tol, use_LeeS=True, delta=delta, verbose=verbose, gamma=1)

    return {"errors" : [error0, error1, error2, error3, error4, error5, error6], 
            "timings" : [toc0, toc1, toc2, toc3, toc4, toc5, toc6],
            "cnt" : [cnt0[::10], cnt1[::10], cnt2[::10], cnt3[::10], cnt4[::10], cnt5[::10], cnt6[::10]]
            }


# -------------------- Post-Processing ------------------- #
import pandas as pd
import shootout.methods.post_processors as pp
pio.templates.default= "plotly_white"

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
ovars = ["mnr"]
df_conv = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars, err_name="errors_interp", time_name="timings_interp")
df_conv = df_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})

# Median plot
df_conv_median_time = pp.median_convergence_plot(df_conv, type="timings", mean=False)

# Convergence plots with all runs
pxfig = px.line(df_conv_median_time, 
            x="timings", 
            y= "errors", 
            color='algorithm',
            facet_row="mnr",
            log_y=True,
            #log_x=True,
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
    width=450*1.62, # in px
    height=450,
    xaxis1=dict(range=[0,1],title_text="Time (s)"),
    xaxis2=dict(range=[0,30]),
    yaxis1=dict(title_text="Fit"),
    yaxis2=dict(title_text="Fit")
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