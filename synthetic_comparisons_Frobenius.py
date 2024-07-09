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

# Personnal comparison toolbox
# you can get it at 
# https://github.com/cohenjer/shootout
from shootout.methods.runners import run_and_track
from shootout.methods.plotters import plot_speed_comparison

plt.close('all')
# --------------------- Choose parameters for grid tests ------------ #
if len(sys.argv)==1 or int(sys.argv[1])==0:
    seeds = [] #no run
    skip=True
else:
    seeds = list(np.arange(int(sys.argv[1])))
    skip=False

variables = {
    "add_track" : {"distribution" : "uniform"},
    "mnr" : [[200,100,5],[1000,400,20]],
    "NbIter" : [200], # for Lee and Seung also
    "NbIter_inner" : 100,
    "SNR" : [100],
    "delta" : 0.1,
    "seed" : seeds,
    "distribution" : "uniform",
    "show_it" : 100,
    "epsilon" : 1e-8,
    "tol" : 0
}

name = "l2_run-10-05-2023"
algs = ["MU_Fro","fastMU_Fro_ex","GD_Fro", "NeNMF_Fro", "HALS", "fastMU_Fro"]

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

    # Initialization for H0 as a random matrix
    Hini = rng.rand(r, n)
    Wini = rng.rand(m, r) #sparse.random(rV, cW, density=0.25).toarray() 
    
    # adding Gaussian noise to the observed data
    N = rng.randn(m,n)
    sigma = 10**(-cfg["SNR"]/20)*np.linalg.norm(Vorig)/np.linalg.norm(N)
    V = Vorig + sigma*N

    # One noise, one init; NMF is not unique and nncvx so we will find several results
    error0, W0, H0, toc0, cnt0 = nmf_f.NMF_Lee_Seung(V,  Wini, Hini, cfg["NbIter"], cfg["NbIter_inner"],tol=cfg["tol"], legacy=False, delta=cfg["delta"], verbose=verbose)
    error2, W2, H2, toc2, cnt2  = nmf_f.Grad_descent(V , Wini, Hini, cfg["NbIter"], cfg["NbIter_inner"], tol=cfg["tol"], delta=cfg["delta"], verbose=verbose)
    error3, W3, H3, toc3, cnt3  = nmf_f.NeNMF(V, Wini, Hini, tol=cfg["tol"], nb_inner=cfg["NbIter_inner"], itermax=cfg["NbIter"], delta=cfg["delta"], verbose=verbose)
    # Fewer max iter because too slow
    
    # With Tensorly --> too slow
    # callback def
    #toc4=[]
    #error4=[]
    #cnt4=[]
    #norm_tensor = tl.norm(V,2)**2
    #def callback_call(cp_tensor,error, inner_iter=None):
        #toc4.append(time.perf_counter())
        #error4.append(np.sqrt(error*2*norm_tensor)/m/n) #err in cp is 1/2 \| \|_F^2
        #if inner_iter is not None:
            #cnt4.append(inner_iter)
    #[W4, H4], _, _ = non_negative_parafac_hals(V, r, init=(None,[np.copy(Wini),np.copy(Hini).T]), n_iter_max=cfg["NbIter"], tol=cfg["tol"], return_errors=True, inner_iter_max=cfg["NbIter_inner"], inner_tol=cfg["delta"]*5, verbose=verbose, callback=callback_call)
    #toc4 = [toc4[i]-toc4[0] for i in range(len(toc4))]
    # Q: error4 is right format?
    
    _, _, error4, toc4, cnt4 = nmf(V, r, init="random", n_iter_max=cfg["NbIter"], tol=cfg["tol"], return_costs=True, NbIter_inner=cfg["NbIter_inner"], delta=cfg["delta"], verbose=verbose)

    error1, W1, H1, toc1, cnt1  = nmf_f.NeNMF_optimMajo(V, Wini, Hini, tol=cfg["tol"], itermax=cfg["NbIter"], nb_inner=cfg["NbIter_inner"], delta=cfg["delta"], verbose=verbose, use_best=False, gamma=1)
    error5, W5, H5, toc5, cnt5 = nmf_f.NMF_proposed_Frobenius(V, Wini, Hini, cfg["NbIter"], cfg["NbIter_inner"], tol=cfg["tol"], delta=cfg["delta"], verbose=verbose, gamma=1.9)

    #   algs = ["MU_Fro","fastMU_Fro_ex","GD_Fro", "NeNMF_Fro", "HALS", "fastMU_Fro"]
    return {"errors" : [error0, error1, error2, error3, error4, error5], 
            "timings" : [toc0, toc1, toc2, toc3, toc4, toc5],
            "cnt" : [cnt0[::10], cnt1[::10], cnt2[::10], cnt3[::10], cnt4[::10], cnt5[::10]]
            }


# -------------------- Post-Processing ------------------- #
import pandas as pd
import shootout.methods.post_processors as pp
pio.templates.default= "plotly_white"

df = pd.read_pickle("Results/"+name)

# Remove extrapolation
df = df[df["algorithm"] != "fastMU_Fro_ex"]
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
df = pp.interpolate_time_and_error(df, npoints = 100, adaptive_grid=True, groups=ovars_interp)

# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
ovars = ["mnr", "SNR","seed"]
df_conv = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars, err_name="errors_interp", time_name="timings_interp")
df_conv = df_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})

# Median plot
df_conv_median_time = pp.median_convergence_plot(df_conv, type_x="timings", mean=False)

# Convergence plots with all runs
pxfig = px.line(
            df_conv_median_time,
            x="timings", 
            y= "errors", 
            color='algorithm',
            line_dash='algorithm',
            facet_col="mnr",
            log_y=True,
            facet_col_spacing=0.1,
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
    xaxis1=dict(range=[0,3],title_text="Time (s)"),
    xaxis2=dict(range=[0,0.2],title_text="Time (s)"),
    yaxis1=dict(title_text="Fit"),
    yaxis2=dict(title_text="")
)

pxfig.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfig.update_yaxes(
    matches=None,
    showticklabels=True
)
# updating titles
for i,ann in enumerate(pxfig.layout.annotations):
    if ann.text[:3]=="mnr":
        ann.text="[m,n,r]"+ann.text[3:] 

pxfig.write_image("Results/"+name+".pdf")
pxfig.write_image("Results/"+name+".pdf")
pxfig.show()