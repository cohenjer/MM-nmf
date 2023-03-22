import numpy as np
from matplotlib import pyplot as plt
import NLS_KL as nls_kl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from utils import sparsify
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
algs = ["MU_KL", "fastMU_KL", "fastMU_KL_approx"] #"fastMU_KL_trueH_min", "fastMU_KL_trueH", "fastMU_KL_min", "fastMU_KL"]
if len(sys.argv)==1:
    nb_seeds = 0 #no run
else:
    nb_seeds = int(sys.argv[1])  # Change this to >0 to run experiments

name = "KL_nls_sparse_run_02-14-2023" #check 05
@run_and_track(algorithm_names=algs, path_store="Results/", name_store=name,
                add_track = {"distribution" : "uniform"},
                nb_seeds=nb_seeds,
                mnr = [[200,100,5],[2000,1000,40]],
                NbIter = [300], # for Lee and Seung also
                SNR = [100],#, 30],
                delta = 0,
                seeded_fun=True,
                setup = ["dense","fac sparse","fac data sparse","data sparse"]
                )
def one_run(mnr=[100,100,5],SNR=50, NbIter=3000, verbose=False, show_it=100, delta=0, seed=1, epsilon=1e-8, setup=1):
    m, n, r = mnr
    # Fixed the signal 
    rng = np.random.RandomState(seed+20)
    Worig = rng.rand(m, r)
    Horig = rng.rand(r, n)
    # Sparsifying
    # Setup 1
    match setup:
        case "dense":# Dense
            Vorig = Worig.dot(Horig) # densified
        case "fac sparse":# sparse factors, dense data
            Worig = sparsify(Worig, s=0.5, epsilon=epsilon)
            Horig = sparsify(Horig, s=0.5, epsilon=epsilon)
            Vorig = Worig.dot(Horig) + 0.01 # densified
        case "fac data sparse":# sparse factors and data
            Worig = sparsify(Worig, s=0.5, epsilon=epsilon)
            Horig = sparsify(Horig, s=0.5, epsilon=epsilon)
            Vorig = Worig.dot(Horig) #+ 0.1 # densified
            Vorig = sparsify(Vorig, s=0.5, epsilon=r*epsilon**2)
        case "data sparse":# dense factors, sparse data
            Vorig = Worig.dot(Horig) #+ 0.1 # densified
            Vorig = sparsify(Vorig, s=0.5, epsilon=r*epsilon**2)

    # Sparsifying
    #print(f"Estce que {np.min(Vorig)} et {r*epsilon**2} sont égaux")

    # Initialization for H0 as a random matrix
    Hini = rng.rand(r, n)
    
    # adding Poisson noise to the observed data
    #N = np.random.poisson(1,size=Vorig.shape) # integers
    N = rng.rand(m,n) # uniform
    sigma = 10**(-SNR/20)*np.linalg.norm(Vorig)/np.linalg.norm(N)
    V = Vorig #+ sigma*N

    # One noise, one init; NMF is not unique and nncvx so we will find several results
    _, Hxx, _ = nls_kl.Lee_Seung_KL(V, Worig, Hini, NbIter=50, verbose=verbose, print_it=show_it, delta=delta, epsilon=epsilon)
    error0, H0, toc0 = nls_kl.Lee_Seung_KL(V, Worig, Hini, NbIter=NbIter, verbose=verbose, print_it=show_it, delta=delta, epsilon=epsilon)
    error1, H1, toc1 = nls_kl.Proposed_KL(V, Worig, Hini, NbIter=NbIter, verbose=verbose, print_it=show_it, delta=delta, use_LeeS=False, gamma=1.9, epsilon=epsilon, true_hessian=True)
    error2, H2, toc2 = nls_kl.Proposed_KL(V, Worig, Hini, NbIter=NbIter, verbose=verbose, print_it=show_it, delta=delta, use_LeeS=False, gamma=1.9, epsilon=epsilon, true_hessian=False)
    #error3, H3, toc3 = nls_kl.Proposed_KL(V, Worig, Hini, NbIter=NbIter, verbose=verbose, print_it=show_it, delta=delta, use_LeeS=True, gamma=1, epsilon=epsilon, true_hessian=False)
    #error4, H4, toc4 = nls_kl.Proposed_KL(V, Worig, Hini, NbIter=NbIter, verbose=verbose, print_it=show_it, delta=delta, use_LeeS=False, gamma=1.9, epsilon=epsilon, true_hessian=False)

    return {
        "errors" : [error0, error1, error2],#,error3,error4], 
        "timings" : [toc0, toc1, toc2],#,toc3,toc4]
            }


# -------------------- Post-Processing ------------------- #
import pandas as pd
pio.templates.default= "plotly_white"

df = pd.read_pickle("Results/"+name)
nb_seeds = df["seed"].max()+1 # get nbseed from data


# Grouping columns
df = pp.regroup_columns(df, keys=["mnr"], how_many=3)
# Making a convergence plot dataframe
df = pp.interpolate_time_and_error(df, npoints = 100, adaptive_grid=True)

#for setup in ["dense","fac sparse","fac data sparse","data sparse"]:
# We will show convergence plots for various sigma values, with only n=100
ovars = ["mnr", "SNR", "setup"]
df_conv = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars, err_name="errors_interp", time_name="timings_interp", filters=dict({"mnr": "[2000, 1000, 40]"}))
df_conv = df_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})

# Median plot
df_conv_median_time = pp.median_convergence_plot(df_conv, type="timings")

# Convergence plots with all runs
pxfig = px.line(df_conv_median_time, 
            x="timings", 
            y= "errors", 
            color='algorithm',
            line_dash='algorithm',
            #facet_row="SNR",
            facet_col="setup",
            facet_col_wrap=2,
            log_y=True,
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
    font_size = 12,
    #title_text = f"NLS Results for Setup {setup}",
    width=600*1.62, # in px
    height=600,
    xaxis1=dict(range=[0,5], title_text="Time (s)"),
    xaxis3=dict(range=[0,5]),
    xaxis2=dict(range=[0,15],title_text="Time (s)"),
    xaxis4=dict(range=[0,15]),
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