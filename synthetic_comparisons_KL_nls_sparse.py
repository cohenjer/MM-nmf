import numpy as np
from matplotlib import pyplot as plt
import NLS_KL as nls_kl
import pandas as pd
import plotly.express as px
import sys
from utils import sparsify, opt_scaling
import plotly.io as pio
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
    "add_track" : {"distribution" : "uniform"},
    "mnr" : [[200,100,10]],
    "NbIter" : [300], # for Lee and Seung also
    "NbIter_SN": [100],
    "SNR" : [100, 20],  #5000 and 50 photons
    "delta" : 0,
    "setup" : ["dense", "sparse"],  #["dense","fac sparse","fac data sparse","data sparse"],
    "seed" : seeds, 
    "distribution" : "uniform",
    "show_it" : 100,
    "epsilon" : 1e-16,
}

algs = ["MU", "mSOM", "MUSOM", "SN CCD"]  #,"SUM Convergent"]
name = "KL_nls_sparse_run_14-04-2025"

@run_and_track(
        algorithm_names = algs, name_store=name,
        path_store="Results/", skip=skip, verbose=True, **variables
        )
def one_run(verbose=True, **cfg):
    #mnr=[100,100,5],SNR=50, NbIter=3000, verbose=False, show_it=100, delta=0, seed=1, epsilon=1e-8, setup=1, skip=False):
    m, n, r = cfg["mnr"]
    # Fixed the signal 
    rng = np.random.RandomState(cfg["seed"]+20)
    Worig = rng.rand(m, r)
    Horig = rng.rand(r, n)
    # Sparsifying
    # Setup 1
    match cfg["setup"]:
        case "dense":  # Dense
            Vorig = Worig.dot(Horig)  # densified
        #case "fac sparse":# sparse factors, dense data
        #    Worig = sparsify(Worig, s=0.5, epsilon=cfg["epsilon"])
        #    Horig = sparsify(Horig, s=0.5, epsilon=cfg["epsilon"])
        #    Vorig = Worig.dot(Horig) + 0.01 # densified
        case "sparse":  # sparse factors and data
            Worig = sparsify(Worig, s=0.5, epsilon=cfg["epsilon"])
            Horig = sparsify(Horig, s=0.5, epsilon=cfg["epsilon"])
            Vorig = Worig.dot(Horig)  #+ 0.1 # densified
            #Vorig = sparsify(Vorig, s=0.5, epsilon=r*cfg["epsilon"]**2)
        #case "data sparse":# dense factors, sparse data e.g. missing data --> non sense ?
            #Vorig = Worig.dot(Horig) #+ 0.1 # densified
            #Vorig = sparsify(Vorig, s=0.5, epsilon=r*cfg["epsilon"]**2)

    # Sparsifying
    #print(f"Estce que {np.min(Vorig)} et {r*epsilon**2} sont égaux")
    
    # adding Poisson noise to the observed data
    #N = np.random.poisson(1,size=Vorig.shape) # integers
    #N = rng.rand(m,n) # uniform
    #sigma = 10**(-cfg["SNR"]/20)*np.linalg.norm(Vorig)/np.linalg.norm(N)
    #V = Vorig + sigma*N
    
    # Generating data with Poisson distribution
    sigma = 0.5*10**(cfg["SNR"]/10)  # intensity for the target SNR, mean x value 0.5
    V = np.random.poisson(sigma*Vorig)
    V = V/np.max(V) # [0,1] normalization
    
    # Initialization for H0 as a random matrix
    Hini = rng.rand(r, n)  # TODO better init ?
    # scaling
    lamb = opt_scaling(V, Worig@Hini)
    Hini = np.maximum(lamb*Hini, cfg["epsilon"])
    #print(lamb)
    
    # scaled nn
    #Hini = nearest_neighbour_H(V, Worig, epsilon = cfg["epsilon"])
    
    #Hini = absls(V, Worig, epsilon = cfg["epsilon"])
    
    
    #from IPython import embed; embed()

    #_, Hxx, _ = nls_kl.Lee_Seung_KL(V, Worig, Hini, NbIter=50, verbose=verbose, print_it=cfg["show_it"], delta=cfg["delta"], epsilon=cfg[ "epsilon" ])
    
    # MU
    error0, H0, toc0 = nls_kl.Lee_Seung_KL(V, Worig, Hini, NbIter=cfg["NbIter"], verbose=verbose, print_it=cfg["show_it"], delta=cfg["delta"], epsilon=cfg["epsilon"])
    
    # mSOM
    error1, H1, toc1 = nls_kl.Proposed_KL(V, Worig, Hini, NbIter=cfg["NbIter"], verbose=verbose, print_it=cfg["show_it"], delta=cfg["delta"], gamma=1.9, method="mSOM", epsilon=cfg["epsilon"])
    
    # MuSOM
    error2, H2, toc2 = nls_kl.Proposed_KL(V, Worig, Hini, NbIter=cfg["NbIter"], verbose=verbose, print_it=cfg["show_it"], delta=cfg["delta"], gamma=1.9, method="MUSOM", epsilon=cfg["epsilon"])
    
    # SN CCD (init scaled comme les autres)
    error3, H3, toc3 = nls_kl.ScalarNewton(V, Worig, Hini, NbIter=cfg["NbIter_SN"], verbose=verbose, print_it=cfg["show_it"], delta=cfg["delta"], method="CCD")
    
    
    #error4, H4, toc4 = nls_kl.Proposed_KL(V, Worig, Hini, NbIter=cfg["NbIter"], verbose=verbose, print_it=cfg["show_it"], delta=cfg["delta"], gamma=1., epsilon=cfg["epsilon"]) # Check cost maj, this one has guarantees with SUM TODO check gamma
    
    #error2, H2, toc2 = nls_kl.GD_KL(V, Worig, Hini, NbIter=cfg["NbIter"], verbose=verbose, print_it=cfg["show_it"], delta=cfg["delta"], epsilon=cfg["epsilon"])
    #error2, H2, toc2 = nls_kl.Proposed_KL(V, Worig, Hini, NbIter=NbIter, verbose=verbose, print_it=show_it, delta=delta, use_LeeS=False, gamma=1.9, epsilon=epsilon, true_hessian=False)
    #error3, H3, toc3 = nls_kl.Proposed_KL(V, Worig, Hini, NbIter=NbIter, verbose=verbose, print_it=show_it, delta=delta, use_LeeS=True, gamma=1, epsilon=epsilon, true_hessian=False)
    #error4, H4, toc4 = nls_kl.Proposed_KL(V, Worig, Hini, NbIter=NbIter, verbose=verbose, print_it=show_it, delta=delta, use_LeeS=False, gamma=1.9, epsilon=epsilon, true_hessian=False)

    return {
        "errors": [error0, error1, error2, error3],  #, error4], 
        "timings": [toc0, toc1, toc2, toc3]  #, toc4]
            }


# -------------------- Post-Processing ------------------- #
import pandas as pd
pio.templates.default= "plotly_white"

df = pd.read_pickle("Results/"+name)

# Grouping columns
#df = pp.regroup_columns(df, keys=["mnr"], how_many=3)
# Making a convergence plot dataframe
ovars_interp = ["mnr", "setup", "algorithm", "SNR"]
df = pp.interpolate_time_and_error(df, npoints = df["NbIter"][0], adaptive_grid=True, groups=ovars_interp)

#for setup in ["dense","fac sparse","fac data sparse","data sparse"]:
# We will show convergence plots for various sigma values, with only n=100
ovars = ["mnr", "SNR", "setup", "seed"]
df_conv = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars, err_name="errors_interp", time_name="timings_interp")
df_conv = df_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
df_conv_it = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars)#, filters=dict({"mnr": "[2000, 1000, 40]"}))

# Median plot
df_conv_median_time = pp.median_convergence_plot(df_conv, type_x="timings")
df_conv_median_it = pp.median_convergence_plot(df_conv_it)

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
    #title_text = f"NLS Results for Setup {setup}",
    width=450, # in px
    height=350,
    #xaxis1=dict(range=[0, 0.2], title_text="Time (s)"),
    #xaxis3=dict(range=[0, 0.2]),
    #xaxis2=dict(range=[0, 0.2], title_text="Time (s)"),
    #xaxis4=dict(range=[0, 0.2]),
    xaxis1=dict(title_text="Time (s)"),
    xaxis2=dict(title_text="Time (s)"),
    yaxis1=dict(title_text="Loss"),
    yaxis3=dict(title_text="Loss")
)

pxfig.update_xaxes(
    matches = None,
    #showticklabels = True
)
pxfig.update_yaxes(
    matches=None,
    showticklabels=True
)

# Convergence plots with all runs
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
    #title_text = f"NLS Results for Setup {setup}",
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