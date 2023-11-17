import numpy as np
from matplotlib import pyplot as plt
import NMF_KL as nmf_kl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import plotly.io as pio
from utils import sparsify
pio.kaleido.scope.mathjax = None

# Personnal comparison toolbox
# you can get it at 
# https://github.com/cohenjer/shootout
from shootout.methods.runners import run_and_track
import shootout.methods.post_processors as pp
from shootout.methods.plotters import plot_speed_comparison

plt.close('all')

# --------------------- Choose parameters for grid tests ------------ #
if len(sys.argv)==1 or not sys.argv[1]:
    seeds = [] #no run
    skip=True
else:
    seeds = list(np.arange(int(sys.argv[1])))
    skip=False

variables = {
    "add_track" : {"distribution" : "uniform"},
    "mnr" : [[200,100,5]],
    "NbIter_inner" : [100],
    "NbIter" : 100, 
    "SNR" : [100],
    "delta" : 0.1,
    "setup" : ["dense","fac sparse","fac data sparse","data sparse"],
    "epsilon" : 1e-8,
    "show_it" : 100,
    "tol": 0, 
    "seed" : seeds,
}

algs = ["MU_KL", "fastMU_KL"]
name = "KL_sparse_run_10-05-2023"

@run_and_track(algorithm_names=algs, path_store="Results/", name_store=name, verbose=True, skip=skip, **variables)
def one_run(verbose=False, **cfg):
    m, n, r = cfg["mnr"]
    # Fixed the signal 
    rng = np.random.RandomState(cfg["seed"]+20)
    Worig = rng.rand(m, r) 
    Horig = rng.rand(r, n)  
    Vorig = Worig.dot(Horig)

    # Initialization for H0 as a random matrix
    Hini = rng.rand(r, n)
    Wini = rng.rand(m, r) #sparse.random(rV, cW, density=0.25).toarray() 
    
    match cfg["setup"]:
        case "dense":# Dense
            Vorig = Worig.dot(Horig) # densified
        case "fac sparse":# sparse factors, dense data
            Worig = sparsify(Worig, s=0.5, epsilon=cfg["epsilon"])
            Horig = sparsify(Horig, s=0.5, epsilon=cfg["epsilon"])
            Vorig = Worig.dot(Horig) #+ 0.01 # densified
        case "fac data sparse":# sparse factors and data
            Worig = sparsify(Worig, s=0.5, epsilon=cfg["epsilon"])
            Horig = sparsify(Horig, s=0.5, epsilon=cfg["epsilon"])
            Vorig = Worig.dot(Horig) #+ 0.1 # densified
            Vorig = sparsify(Vorig, s=0.5, epsilon=r*cfg["epsilon"]**2)
        case "data sparse":# dense factors, sparse data
            Vorig = Worig.dot(Horig) #+ 0.1 # densified
            Vorig = sparsify(Vorig, s=0.5, epsilon=r*cfg["epsilon"]**2)

    # adding Poisson noise to the observed data
    #N = np.random.poisson(1,size=Vorig.shape) # integers
    N = rng.rand(m,n) # uniform
    sigma = 10**(-cfg["SNR"]/20)*np.linalg.norm(Vorig)/np.linalg.norm(N)
    V = Vorig + sigma*N

    # One noise, one init; NMF is not unique and nncvx so we will find several results
    error0, W0, H0, toc0, cnt0 = nmf_kl.Lee_Seung_KL(V, Wini, Hini, NbIter=cfg["NbIter"], nb_inner=cfg["NbIter_inner"], tol=cfg["tol"], verbose=verbose, print_it=cfg["show_it"], delta=cfg["delta"])
    error1, W1, H1, toc1, cnt1 = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=cfg["NbIter"], nb_inner=cfg["NbIter_inner"], tol=cfg["tol"], verbose=verbose, print_it=cfg["show_it"], delta=cfg["delta"], gamma=1.9, epsilon=cfg["epsilon"])
    #error2, W2, H2, toc2, cnt2 = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=cfg["NbIter_inner"], tol=tol, verbose=verbose, print_it=show_it, delta=delta, use_LeeS=False, gamma=1.9, true_hessian=False, epsilon=epsilon)
    #error3, W3, H3, toc3, cnt3 = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it, delta=delta, use_LeeS=True, gamma=1, true_hessian=False, epsilon=epsilon)
    #error4, W4, H4, toc4, cnt4 = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it, delta=delta, use_LeeS=False, gamma=1.9, true_hessian=False, epsilon=epsilon)

    return {"errors" : [error0, error1],
            "timings" : [toc0, toc1],
            "cnt" :  [cnt0[::10], cnt1[::10]],
           }


# -------------------- Post-Processing ------------------- #
import pandas as pd
pio.templates.default= "plotly_white"

df = pd.read_pickle("Results/"+name)
nb_seeds = df["seed"].max()+1 # get nbseed from data

# Making a convergence plot dataframe
ovars_interp = ["mnr", "setup", "algorithm"]
df = pp.interpolate_time_and_error(df, npoints = 100, adaptive_grid=True, groups=ovars_interp)

#for setup in ["dense","fac sparse","fac data sparse","data sparse"]:
# We will show convergence plots for various sigma values, with only n=100
ovars = ["mnr", "setup"]
df_conv = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars,err_name="errors_interp", time_name="timings_interp")#, filters=dict({"setup":setup}))
df_conv = df_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})

# Median plot
df_conv_median_time = pp.median_convergence_plot(df_conv, type_x="timings")

# Convergence plots with all runs
pxfig = px.line(df_conv_median_time, 
            x="timings", 
            y= "errors", 
            color='algorithm',
            line_dash='algorithm',
            #facet_row="mnr",
            facet_col="setup",
            facet_col_wrap=2,
            log_y=True,
            facet_col_spacing=0.1,
            facet_row_spacing=0.1,
            #error_y="q_errors_p", 
            #error_y_minus="q_errors_m" 
)

# Final touch
pxfig.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfig.update_layout(
    font_size = 12,
    #title_text = f"NMF Results for Setup {setup}",
    width=600*1.62, # in px
    height=600,
    xaxis1=dict(range=[0,0.2], title_text="Time (s)"),
    xaxis2=dict(range=[0,0.3], title_text="Time (s)"),
    xaxis3=dict(range=[0,0.2]),
    xaxis4=dict(range=[0,0.3]),
    yaxis1=dict(title_text="Fit"),
    yaxis3=dict(title_text="Fit")
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
#pxfig.write_image("Results/"+name+"_"+setup+".pdf")
#pxfig.write_image("Results/"+name+"_"+setup+".pdf")
#pxfig.show()
