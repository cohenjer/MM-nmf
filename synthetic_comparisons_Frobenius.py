import numpy as np
from matplotlib import pyplot as plt
import NMF_Frobenius as nmf_f 
from nn_fac.nmf import nmf
#import tensorly as tl #perso branch
#from tensorly.decomposition import non_negative_parafac_hals
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import plotly.io as pio
import time
pio.kaleido.scope.mathjax = None
from utils import opt_scaling_fro

# Personnal comparison toolbox
# you can get it at 
# https://github.com/cohenjer/shootout
from shootout.methods.runners import run_and_track

plt.close('all')
# --------------------- Choose parameters for grid tests ------------ #
if len(sys.argv)==1 or int(sys.argv[1])==0:
    seeds = [] #no run
    skip=True
else:
    seeds = list(np.arange(int(sys.argv[1])))
    skip=False

variables = {
    "add_track": {"distribution" : "uniform"},
    "mnr": [[200, 100, 10], [1000, 400, 20]],
    "NbIter": [200],  # for Lee and Seung also
    "NbIter_HALS": [100],
    "NbIter_inner": 10,
    "SNR": [100, 40],
    "delta": 0,
    "seed": seeds,
    "distribution": "uniform",
    "show_it": 100,
    "epsilon": 1e-8,
    "tol": 0
}

name = "l2_run-14-04-2025"
algs = ["AMU", "APGD", "ANeNMF", "AHALS", "AmSOM", "AMUSOM"]

@run_and_track(algorithm_names=algs, path_store="Results/", name_store=name,
                skip=skip, **variables)
def one_run(**cfg):
    m, n, r = cfg["mnr"]
    # Fixed the signal 
    rng = np.random.RandomState(cfg["seed"]+20)
    Worig = rng.rand(m, r) 
    Horig = rng.rand(r, n)  
    Vorig = Worig.dot(Horig)

    # prints
    verbose = True
    
    # adding Gaussian noise to the observed data
    N = rng.randn(m,n)
    sigma = 10**(-cfg["SNR"]/20)*np.linalg.norm(Vorig)/np.linalg.norm(N)
    V = Vorig + sigma*N

    # Initialization for H0 as a random matrix
    Hini = rng.rand(r, n)
    Wini = rng.rand(m, r) #sparse.random(rV, cW, density=0.25).toarray() 
    Hini = opt_scaling_fro(V, Wini@Hini)*Hini
    
    # One noise, one init; NMF is not unique and nncvx so we will find several results
    error0, W0, H0, toc0, cnt0 = nmf_f.NMF_Lee_Seung(V,  Wini, Hini, cfg["NbIter"], cfg["NbIter_inner"],tol=cfg["tol"], legacy=False, delta=cfg["delta"], verbose=verbose)
    error2, W2, H2, toc2, cnt2  = nmf_f.Grad_descent(V , Wini, Hini, cfg["NbIter"], cfg["NbIter_inner"], tol=cfg["tol"], delta=cfg["delta"], verbose=verbose)
    error3, W3, H3, toc3, cnt3  = nmf_f.NeNMF(V, Wini, Hini, tol=cfg["tol"], nb_inner=cfg["NbIter_inner"], itermax=cfg["NbIter"], delta=cfg["delta"], verbose=verbose)
    # Fewer max iter because too slow
    _, _, error4, toc4, cnt4 = nmf(V, r, init="random", n_iter_max=cfg["NbIter_HALS"], tol=cfg["tol"], return_costs=True, NbIter_inner=cfg["NbIter_inner"], delta=cfg["delta"], verbose=verbose)

    error5, W5, H5, toc5, cnt5 = nmf_f.NMF_proposed_Frobenius(V, Wini, Hini, cfg["NbIter"], cfg["NbIter_inner"], tol=cfg["tol"], delta=cfg["delta"], verbose=verbose, gamma=1.9, method="AmSOM")
    error6, W6, H6, toc6, cnt6 = nmf_f.NMF_proposed_Frobenius(V, Wini, Hini, cfg["NbIter"], cfg["NbIter_inner"], tol=cfg["tol"], delta=cfg["delta"], verbose=verbose, gamma=1.9, method="AMUSOM")

    #   algs = ["MU_Fro","fastMU_Fro_ex","GD_Fro", "NeNMF_Fro", "HALS", "fastMU_Fro", "trueMU_Fro"]
    return {"errors" : [error0, error2, error3, error4, error5, error6], 
            "timings" : [toc0, toc2, toc3, toc4, toc5, toc6],
            "cnt" : [cnt0[::10], cnt2[::10], cnt3[::10], cnt4[::10], cnt5[::10], cnt6[::10]]
            }


# -------------------- Post-Processing ------------------- #
import pandas as pd
import shootout.methods.post_processors as pp
pio.templates.default= "plotly_white"

df = pd.read_pickle("Results/"+name)

# Remove extrapolation
#df = df[df["algorithm"] != "fastMU_Fro_ex"]
# TODO: shootout plots
## Using shootout for plotting and postprocessing
#thresh = np.logspace(-3,-8,50) 
#scores_time, scores_it, timings, iterations = pp.find_best_at_all_thresh(df,thresh, nb_seeds)

# ----------------------- Plot --------------------------- #
#fig_winner = plot_speed_comparison(thresh, scores_time, scores_it, legend=algs)
#fig_winner.show()

# Adding in results errors at specific timings and iterations
#df = pp.error_at_time_or_it(df, time_stamps=[0.1, 0.5, 1], it_stamps=[10, 50, 300])

# Group up columns
#df = pp.regroup_columns(df, keys=["mnr"], how_many=3)

# Interpolating time (choose fewer points for better vis), adaptive grid since time varies across plots
ovars_interp =  ["mnr", "SNR", "algorithm"]
df = pp.interpolate_time_and_error(df, npoints=df["NbIter"][0], adaptive_grid=True, groups=ovars_interp)

# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
ovars = ["mnr", "SNR", "seed"]
df_conv = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars, err_name="errors_interp", time_name="timings_interp")
df_conv = df_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
df_conv_it = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars)

# Median plot
df_conv_median_time = pp.median_convergence_plot(df_conv, type_x="timings", mean=False)
df_conv_median_it = pp.median_convergence_plot(df_conv, mean=False)

# Convergence plots with all runs
pxfig = px.line(
            df_conv_median_time,
            x="timings", 
            y= "errors", 
            color='algorithm',
            line_dash='algorithm',
            facet_col="mnr",
            facet_row="SNR",
            log_y=True,
            log_x=True,
            facet_col_spacing=0.1,
            facet_row_spacing=0.1,
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
    font_size = 10,
    width=450, # in px
    height=350,
    xaxis1=dict(title_text="Time (s)"),
    xaxis2=dict(title_text="Time (s)"),
    #xaxis1=dict(range=[0,1.5],title_text="Time (s)"),
    #xaxis2=dict(range=[0,0.25],title_text="Time (s)"),
    #xaxis3=dict(range=[0,1.5]),
    #xaxis4=dict(range=[0,0.25]),
    yaxis1=dict(title_text="n. Loss"),
    yaxis2=dict(title_text=""),
    yaxis3=dict(title_text="n. Loss"),
    yaxis4=dict(title_text="")
)

pxfig.update_xaxes(
    matches = None,
    #showticklabels = True
)
pxfig.update_yaxes(
    matches=None,
    showticklabels=True
)
# updating titles
for i,ann in enumerate(pxfig.layout.annotations):
    if ann.text[:3]=="mnr":
        ann.text="[M,N,R]"+ann.text[3:] 

# Convergence plots with all runs
pxfigit = px.line(
            df_conv_median_it,
            x="it", 
            y= "errors", 
            color='algorithm',
            line_dash='algorithm',
            facet_col="mnr",
            facet_row="SNR",
            log_y=True,
            facet_col_spacing=0.1,
            facet_row_spacing=0.1,
            #log_x=True,
            #error_y="q_errors_p", 
            #error_y_minus="q_errors_m", 
)
# Final touch
pxfigit.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfigit.update_layout(
    font_size = 10,
    width=450, # in px
    height=350,
    #xaxis1=dict(range=[0,3],title_text="Time (s)"),
    #xaxis2=dict(range=[0,0.2],title_text="Time (s)"),
    yaxis1=dict(title_text="n. Loss"),
    yaxis2=dict(title_text=""),
    yaxis3=dict(title_text="n. Loss"),
    yaxis4=dict(title_text="")
)

pxfigit.update_xaxes(
    matches = None,
    #showticklabels = True
)
pxfigit.update_yaxes(
    matches=None,
    showticklabels=True
)
# updating titles
for i,ann in enumerate(pxfigit.layout.annotations):
    if ann.text[:3]=="mnr":
        ann.text="[M,N,R]"+ann.text[3:] 
pxfig.write_image("Results/"+name+".pdf")
pxfig.write_image("Results/"+name+".pdf")
pxfigit.write_image("Results/"+name+"_it.pdf")
pxfig.show()
pxfigit.show()