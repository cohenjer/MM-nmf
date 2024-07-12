import numpy as np
from matplotlib import pyplot as plt
import NLS_Frobenius as nls_f 
import nn_fac
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from utils import opt_scaling_fro
import sys
import plotly.io as pio
pio.kaleido.scope.mathjax = None

#ToChange for new shootout version:
# - nbseed -> seeds and skip
# - type_x for median

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
    "add_track" : {"distribution" : "uniform"},
    "mnr" : [[200,100,5], [1000,400,20]],
    "NbIter" : [200], # for Lee and Seung also
    "SNR" : [100, 30],
    "delta" : 0,
    "seed" : seeds,
    "distribution" : "uniform",
    "show_it" : 100,
    "epsilon" : 1e-8
}

algs = ["MU_Fro","fastMU_Fro_ex","GD_Fro", "NeNMF_Fro", "HALS", "fastMU_Fro", "trueMU_Fro"]
name = "l2_nls_run-12-07-2024"

@run_and_track(algorithm_names=algs, path_store="Results/", name_store=name, skip=skip,
               **variables)
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
    N = rng.randn(m, n)
    sigma = 10**(-cfg["SNR"]/20)*np.linalg.norm(Vorig)/np.linalg.norm(N)
    V = Vorig + sigma*N
    
    # Initialization for H0 as a random matrix
    Hini = rng.rand(r, n)
    Hini = opt_scaling_fro(V, Worig@Hini)*Hini  # TODO CHANGE PAPER
    
    # One noise, one init; NMF is not unique and nncvx so we will find several results

    # Baselines
    error0, H0, toc0 = nls_f.NMF_Lee_Seung(V,  Worig, Hini, cfg["NbIter"], legacy=False, delta=cfg["delta"], verbose=verbose)
    error2, H2, toc2  = nls_f.Grad_descent(V , Worig, Hini, cfg["NbIter"], delta=cfg["delta"], verbose=verbose)
    error3, H3, toc3  = nls_f.NeNMF(V, Worig, Hini, itermax=cfg["NbIter"], delta=cfg["delta"], verbose=verbose)
    
    # HALS is unfair because we compute things before. We add the time needed for this back after the algorithm
    tic = time.perf_counter()
    WtV = Worig.T@V
    WtW = Worig.T@Worig
    toc4_offset = time.perf_counter() - tic
    H4, _, _, _, error4, toc4 = nn_fac.nnls.hals_nnls_acc(WtV, WtW, np.copy(Hini), maxiter=cfg["NbIter"], return_error=True, delta=cfg["delta"], M=V)
    toc4 = [toc4[i] + toc4_offset for i in range(len(toc4))] # leave the 0 in place for init
    toc4[0]=0

    # Proposed methods
    error1, H1, toc1  = nls_f.NeNMF_optimMajo(V, Worig, Hini, itermax=cfg["NbIter"], delta=cfg["delta"], verbose=verbose, gamma=1)
    error5, H5, toc5 = nls_f.NMF_proposed_Frobenius(V, Worig, Hini, cfg["NbIter"], delta=cfg["delta"], verbose=verbose, method="fastMU")
    error6, H6, toc6 = nls_f.NMF_proposed_Frobenius(V, Worig, Hini, cfg["NbIter"], delta=cfg["delta"], verbose=verbose, method="trueMU")

    return {
            "errors": [error0, error1, error2, error3, error4, error5, error6],
            "timings": [toc0, toc1, toc2, toc3, toc4, toc5, toc6],
           }


# -------------------- Post-Processing ------------------- #
import pandas as pd
import shootout.methods.post_processors as pp
pio.templates.default= "plotly_white"

df = pd.read_pickle("Results/"+name)

# Remove extrapolation
df = df[df["algorithm"] != "fastMU_Fro_ex"]

# Interpolating time (choose fewer points for better vis), adaptive grid since time varies across plots
ovars_inter = ["mnr", "SNR", "algorithm"]
df = pp.interpolate_time_and_error(df, npoints = 100, adaptive_grid=True, groups=ovars_inter)#, strategy="min_curve")

ovars = ["mnr", "SNR", "seed"]
# Making a convergence plot dataframe
# We will show convergence plots for various sigma values, with only n=100
df_conv = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars, err_name="errors_interp", time_name="timings_interp")
df_conv = df_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
df_conv_it = pp.df_to_convergence_df(df, groups=True, groups_names=ovars, other_names=ovars)

# Median plot
df_conv_median_time = pp.median_convergence_plot(df_conv, type_x="timings", mean=False)
df_conv_median_it = pp.median_convergence_plot(df_conv, mean=False) # ca fait quoi mean?

# Convergence plots with all runs
pxfig = px.line(#df_conv_median_time, 
            df_conv_median_time,
            #x="timings", 
            x="timings", 
            y= "errors", 
            color='algorithm',
            line_dash='algorithm',
            facet_col="mnr",
            facet_row="SNR",
            log_y=True,
            facet_col_spacing=0.1,
            facet_row_spacing=0.1,
            #line_group="groups",
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
    xaxis1=dict(range=[0,0.05], title_text="Time (s)"),
    xaxis3=dict(range=[0,0.05]),
    xaxis2=dict(range=[0,0.01], title_text="Time (s)"),
    xaxis4=dict(range=[0,0.005]),
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
# updating titles
for i,ann in enumerate(pxfig.layout.annotations):
    if ann.text[:3]=="mnr":
        ann.text="[m,n,r]"+ann.text[3:] 

# Convergence plots with all runs
pxfigit = px.line(#df_conv_median_time, 
            df_conv_median_it,
            #x="timings", 
            x="it", 
            y= "errors", 
            color='algorithm',
            line_dash='algorithm',
            facet_col="mnr",
            facet_row="SNR",
            log_y=True,
            facet_col_spacing=0.1,
            facet_row_spacing=0.1,
            #line_group="groups",
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
    font_size = 12,
    width=450*1.62, # in px
    height=450,
    #xaxis1=dict(range=[0,0.05], title_text="Iters"),
    #xaxis3=dict(range=[0,0.05]),
    #xaxis2=dict(range=[0,0.01], title_text="Iters"),
    #xaxis4=dict(range=[0,0.005]),
    yaxis1=dict(title_text="Fit"),
    yaxis3=dict(title_text="Fit")
)

pxfigit.update_xaxes(
    matches = None,
    showticklabels = True
)
pxfigit.update_yaxes(
    matches=None,
    showticklabels=True
)
# updating titles
for i,ann in enumerate(pxfigit.layout.annotations):
    if ann.text[:3]=="mnr":
        ann.text="[m,n,r]"+ann.text[3:] 



# we save twice because of kaleido+browser bug...
pxfig.write_image("Results/"+name+".pdf")
pxfig.write_image("Results/"+name+".pdf")
pxfigit.write_image("Results/"+name+"_it.pdf")
pxfig.show()
pxfigit.show()