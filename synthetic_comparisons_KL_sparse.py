import numpy as np
from matplotlib import pyplot as plt
import NMF_KL as nmf_kl
import pandas as pd
import plotly.express as px
import sys
import plotly.io as pio
from utils import sparsify, opt_scaling
pio.kaleido.scope.mathjax = None

# Personnal comparison toolbox
# you can get it at 
# https://github.com/cohenjer/shootout
from shootout.methods.runners import run_and_track
import shootout.methods.post_processors as pp
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
    "add_track": {"distribution": "uniform"},
    "mnr": [[200, 100, 10]],
    "NbIter_inner": [10],
    "NbIter_inner_SN": [3],
    "NbIter": [100], 
    "NbIter_SN": [20],
    # Testing [40,20]
    "SNR": [100, 20],  #5000 and 50 photons
    "delta": 0, 
    "setup": ["dense", "sparse"],
    "epsilon": 1e-16,
    "show_it": 100,
    "tol": 0, 
    "seed": seeds,
}

algs = ["AMU", "AmSOM", "AMUSOM", "ASN CCD"]#
name = "KL_sparse_run_14-04-2025"
#name = "trash_KL_sparse_run_14-04-2025"

@run_and_track(algorithm_names=algs, path_store="Results/", name_store=name, verbose=True, skip=skip, **variables)
def one_run(verbose=True, **cfg):
    m, n, r = cfg["mnr"]
    # Fixed the signal 
    rng = np.random.RandomState(cfg["seed"]+20)
    Worig = rng.rand(m, r) 
    Horig = rng.rand(r, n)  
    Vorig = Worig.dot(Horig)

    
    match cfg["setup"]:
        case "dense":  # Dense
            Vorig = Worig.dot(Horig)  # densified
        case "sparse":  # sparse factors and data
            Worig = sparsify(Worig, s=0.5, epsilon=cfg["epsilon"])
            Horig = sparsify(Horig, s=0.5, epsilon=cfg["epsilon"])
            Vorig = Worig.dot(Horig)  #+ 0.1 # densified

    # adding Poisson noise to the observed data
    #N = np.random.poisson(1,size=Vorig.shape) # integers
    #N = rng.rand(m,n) # uniform
    #sigma = 10**(-cfg["SNR"]/20)*np.linalg.norm(Vorig)/np.linalg.norm(N)
    #V = Vorig + sigma*N
    
    # Generating data with Poisson distribution
    sigma = 0.5*10**(cfg["SNR"]/10)# intensity for the target SNR, mean x value 0.5
    # True SNR depends on value of x, very low if x is low. Here is give the average SNR of sorts
    V = np.maximum(np.random.poisson(sigma*Vorig), cfg["epsilon"])
    V = V/np.max(V) # [0,1] normalization

    # Initialization for H0 as a random matrix
    Hini = rng.rand(r, n)
    Wini = rng.rand(m, r)  # sparse.random(rV, cW, density=0.25).toarray() 
    lamb = opt_scaling(V, Wini@Hini)
    Hini = np.maximum(lamb*Hini, cfg["epsilon"])  # Sinkhorn ??
    # REFINEMENT OF INIT
    _, Wini, Hini, _, _ = nmf_kl.Lee_Seung_KL(V, Wini, Hini, NbIter=1, nb_inner=cfg["NbIter_inner"], tol=0, verbose=verbose, print_it=cfg["show_it"], delta=0) # TODO CHANGED
    
    # One noise, one init; NMF is not unique and nncvx so we will find several results
    # MU
    error0, W0, H0, toc0, cnt0 = nmf_kl.Lee_Seung_KL(V, Wini, Hini, NbIter=cfg["NbIter"], nb_inner=cfg["NbIter_inner"], tol=cfg["tol"], verbose=verbose, print_it=cfg["show_it"], delta=cfg["delta"])
    
    # mSOM
    error1, W1, H1, toc1, cnt1 = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=cfg["NbIter"], nb_inner=cfg["NbIter_inner"], tol=cfg["tol"], verbose=verbose, print_it=cfg["show_it"], delta=cfg["delta"], gamma=1.9, method="AmSOM", epsilon=cfg["epsilon"])
    
    # MuSOM
    error2, W2, H2, toc2, cnt2 = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=cfg["NbIter"], nb_inner=cfg["NbIter_inner"], tol=cfg["tol"], verbose=verbose, print_it=cfg["show_it"], delta=cfg["delta"], gamma=1.9, method="AMUSOM", epsilon=cfg["epsilon"])
    
    # SN (CCD)
    error3, W3, H3, toc3, cnt3 = nmf_kl.ScalarNewton(V, Wini, Hini, NbIter=cfg["NbIter_SN"], nb_inner=cfg["NbIter_inner_SN"], tol=cfg["tol"], verbose=verbose, print_it=cfg["show_it"], delta=cfg["delta"], method="CCD", epsilon=cfg["epsilon"])  # TODO care inner stop
    # TODO instabilité SN CDD handled heuristically with epsilons
    
    
    #error4, W4, H4, toc4, cnt4 = nmf_kl.Proposed_KL(V, Wini, Hini, NbIter=NbIter, nb_inner=NbIter_inner, tol=tol, verbose=verbose, print_it=show_it, delta=delta, use_LeeS=False, gamma=1.9, true_hessian=False, epsilon=epsilon)

    return {
            "errors": [error0, error1, error2, error3],
            "timings": [toc0, toc1, toc2, toc3],
            "cnt":  [cnt0[::10], cnt1[::10], cnt2[::10], cnt3[::10]],
           }


# -------------------- Post-Processing ------------------- #
import pandas as pd
pio.templates.default= "plotly_white"

df = pd.read_pickle("Results/"+name)
nb_seeds = df["seed"].max()+1 # get nbseed from data

# Making a convergence plot dataframe
ovars_interp = ["mnr", "setup", "SNR", "algorithm"]
df = pp.interpolate_time_and_error(df, npoints=df["NbIter"][0], adaptive_grid=True, groups=ovars_interp)

#for setup in ["dense","fac sparse","fac data sparse","data sparse"]:
# We will show convergence plots for various sigma values, with only n=100
ovars = ["mnr", "setup", "SNR", "seed"]
df_conv = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars,err_name="errors_interp", time_name="timings_interp")#, filters=dict({"setup":setup}))
df_conv = df_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
df_conv_it = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars)#, filters=dict({"setup":setup}))

# Median plot
df_conv_median_time = pp.median_convergence_plot(df_conv, type_x="timings")
df_conv_median_it = pp.median_convergence_plot(df_conv_it, type_x="iterations")

# Convergence plots with all runs
pxfig = px.line(df_conv_median_time, 
            x="timings", 
            y= "errors", 
            color='algorithm',
            line_dash='algorithm',
            facet_row="SNR",
            facet_col="setup",
            facet_col_wrap=2,
            log_y=True,
            log_x=True,
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
    font_size = 10,
    #title_text = f"NMF Results for Setup {setup}",
    width=450, # in px
    height=350,
    #xaxis1=dict(range=[0,0.5], title_text="Time (s)"),
    #xaxis2=dict(range=[0,0.5], title_text="Time (s)"),
    #xaxis3=dict(range=[0,0.5]),
    #xaxis4=dict(range=[0,0.5]),
    xaxis1=dict(title_text="Time (s)"),
    xaxis2=dict(title_text="Time (s)"),
    yaxis1=dict(title_text="Loss"),
    yaxis3=dict(title_text="Loss")
)

pxfig.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfig.update_yaxes(
    matches=None,
    showticklabels=True
)

# Convergence plots with all runs its
pxfigit = px.line(df_conv_median_it, 
            x="it", 
            y= "errors", 
            color='algorithm',
            line_dash='algorithm',
            facet_row="SNR",
            facet_col="setup",
            facet_col_wrap=2,
            log_y=True,
            facet_col_spacing=0.1,
            facet_row_spacing=0.1,
            #error_y="q_errors_p", 
            #error_y_minus="q_errors_m" 
)

# Final touch
pxfigit.update_traces(
    selector=dict(),
    line_width=2.5,
    #error_y_thickness = 0.3,
)

pxfigit.update_layout(
    font_size = 10,
    #title_text = f"NMF Results for Setup {setup}",
    width=450, # in px
    height=350,
    yaxis1=dict(title_text="Loss"),
    yaxis3=dict(title_text="Loss")
)

pxfigit.update_xaxes(
    matches = None,
    #showticklabels = True
)
pxfigit.update_yaxes(
    matches=None,
    showticklabels=True
)

pxfig.write_image("Results/"+name+".pdf")
pxfig.write_image("Results/"+name+".pdf")
pxfigit.write_image("Results/"+name+"_it.pdf")
pxfig.show()
pxfigit.show()
#pxfig.write_image("Results/"+name+"_"+setup+".pdf")
#pxfig.write_image("Results/"+name+"_"+setup+".pdf")
#pxfig.show()
