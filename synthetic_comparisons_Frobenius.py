import numpy as np
from matplotlib import pyplot as plt
import NMF_Frobenius as nmf_f 
import time
import nn_fac
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import utils

plt.close('all')
# Fixe the matrix sizes

# Seeding
#np.random.seed(hash(´toto'))

# Storing intermediate results
df = pd.DataFrame()

# --------------------- Choose parameters for grid tests ------------ #
# dimensions
m_list = [40]
n_list = [40]
r_list = [2,5,10] #30

# Max number of iterations
NbIter = 3000
NbIter_hals = 1000 
# Fixed number of inner iterations
NbIter_inner_list = [10]
#NbIter_inner_list = [1,2,5,10,20,30]
# Stopping criterion error<tol
tol = 0 #running all 5k iterations

# noise variance TODO change to snr
sigma_list = [0,1e-7,1e-4]

# Number of random inits
NbSeed=10
# -----------------------------------------------------------------------

Error0 = np.zeros(NbSeed)
Error1 = np.zeros(NbSeed)
Error2 = np.zeros(NbSeed)     
Error3 = np.zeros(NbSeed)
Error4 = np.zeros(NbSeed)

NbIterStop0 = np.zeros(NbSeed)
NbIterStop1 = np.zeros(NbSeed)
NbIterStop2 = np.zeros(NbSeed)
NbIterStop3 = np.zeros(NbSeed)
NbIterStop4 = np.zeros(NbSeed)

for s in range(NbSeed): #[NbSeed-1]:#

    # friendly print
    print(s)
        
    for r in r_list:
        for sigma in sigma_list:
            for n in n_list:
                for m in m_list:

                    # Fixed the signal 
                    Worig = np.random.rand(m, r) 
                    Horig = np.random.rand(r, n)  
                    Vorig = Worig.dot(Horig)

                    # Initialization for H0 as a random matrix
                    Hini = np.random.rand(r, n)
                    Wini = np.random.rand(m, r) #sparse.random(rV, cW, density=0.25).toarray() 
                    
                    # adding noise to the observed data
                    np.random.seed(s)
                    N = sigma*np.random.rand(m,n)
                    V = Vorig + N

                    for NbIter_inner in  NbIter_inner_list:
                        # One noise, one init; NMF is not unique and nncvx so we will find several results
                        error0, W0, H0, toc0 = nmf_f.NMF_Lee_Seung(V,  Wini, Hini, NbIter, NbIter_inner,tol=tol, legacy=False)
                        error1, W1, H1, toc1  = nmf_f.NeNMF_optimMajo(V, Wini, Hini, tol=tol, itermax=NbIter, nb_inner=NbIter_inner)
                        #error1, W1, H1, toc1 = nmf_f.NMF_proposed_Frobenius(V, Wini, Hini, NbIter, NbIter_inner, tol=tol)
                        error2, W2, H2, toc2  = nmf_f.Grad_descent(V , Wini, Hini, NbIter, NbIter_inner, tol=tol)
                        error3, W3, H3, toc3  = nmf_f.NeNMF(V, Wini, Hini, tol=tol, nb_inner=NbIter_inner, itermax=NbIter)
                        # Fewer max iter cause too slow
                        W4, H4, error4, toc4 = nn_fac.nmf.nmf(V, r, init="custom", U_0=np.copy(Wini), V_0=np.copy(Hini), n_iter_max=NbIter_hals, tol=tol, update_rule='hals',beta=2, return_costs=True, NbIter_inner=NbIter_inner)

                        # post-processing: finding errors at time=t or iters=it
                        time_stamps = [0.1, 0.5, 1]
                        it_stamps = [10, 50, 300]
                        if np.min([toc0[-1],toc1[-1],toc2[-1],toc3[-1],toc4[-1]])<time_stamps[-1]:
                            print('Warning: time threshold is too large')
                        idx_time_0 = [np.argmin(np.abs(np.add(toc0,-i))) for i in time_stamps]
                        idx_time_1 = [np.argmin(np.abs(np.add(toc1,-i))) for i in time_stamps]
                        idx_time_2 = [np.argmin(np.abs(np.add(toc2,-i))) for i in time_stamps]
                        idx_time_3 = [np.argmin(np.abs(np.add(toc3,-i))) for i in time_stamps]
                        idx_time_4 = [np.argmin(np.abs(np.add(toc4,-i))) for i in time_stamps]
                        
                        err_time_0 = [error0[i] for i in idx_time_0]
                        err_time_1 = [error1[i] for i in idx_time_1]
                        err_time_2 = [error2[i] for i in idx_time_2]
                        err_time_3 = [error3[i] for i in idx_time_3]
                        err_time_4 = [error4[i] for i in idx_time_4]

                        err_it_0 = [error0[np.minimum(i,len(error0)-1)] for i in it_stamps]
                        err_it_1 = [error1[np.minimum(i,len(error1)-1)] for i in it_stamps]
                        err_it_2 = [error2[np.minimum(i,len(error2)-1)] for i in it_stamps]
                        err_it_3 = [error3[np.minimum(i,len(error3)-1)] for i in it_stamps]
                        err_it_4 = [error4[np.minimum(i,len(error4)-1)] for i in it_stamps]

                        dic= {
                                "batch_size": 5, # number of algorithms in each comparison run
                                "method": ["NMF_LeeSeung",  "NeNMF_optimMajorant", "PGD", "NeNMF", "HALS"],
                                "m": m,
                                "n": n,
                                "r": r,
                                "seed_idx": s,
                                "noise_variance": sigma,
                                "NbIter": NbIter,
                                "NbIter_inner": NbIter_inner,
                                "NbIter_hals": NbIter_hals,
                                "tol": tol,
                                "final_error": [error0[-1], error1[-1], error2[-1], error3[-1], error4[-1]],
                                "total_time": [toc0[-1], toc1[-1], toc2[-1], toc3[-1], toc4[-1]],
                                "full_error": [error0,error1,error2,error3,error4],
                                "full_time": [toc0,toc1,toc2,toc3,toc4],
                                "NbIterStop": [len(error0),len(error1),len(error2),len(error3),len(error4)]
                            }
                        for thres_idx,thres in enumerate(time_stamps):
                            dic["error_at_time_"+str(thres)] = [err_time_0[thres_idx],err_time_1[thres_idx],err_time_2[thres_idx],err_time_3[thres_idx],err_time_4[thres_idx]]
                        for it_idx,it in enumerate(it_stamps):
                            dic["error_at_it_"+str(it)] = [err_it_0[it_idx],err_it_1[it_idx],err_it_2[it_idx],err_it_3[it_idx],err_it_4[it_idx]]

                        df = pd.concat([df,pd.DataFrame(dic)], ignore_index=True)


# Winner at given threshold plots
thresh = np.logspace(-3,-8,50) 
scores_time, scores_it, timings, iterations = utils.find_best_at_all_thresh(df,thresh, 5)

#fig_winner = make_subplots(rows=1,cols=2)
#for i in range(scores_time.shape[0]):
    #fig_winner.add_trace(
        #go.Scatter(x = thresh, y=scores_time[i,:]), row=1,col=1
    #)
    #fig_winner.add_trace(
        #go.Scatter(x = thresh, y=scores_it[i,:]), row=1,col=2
    #)
##fig_winner.update_layout(height=600, width=800, title_text="Side By Side Subplots")
#fig_winner.show()

fig0 = plt.figure()
plt.subplot(121)
plt.semilogx(thresh, scores_time.T)
plt.legend(["NMF_LeeSeung",  "NeNMF_optimMajorant", "PGD", "NeNMF", "HALS"])
plt.title('How many times each algorithm reached threshold the fastest (time)')
plt.xlabel('Rec error threshold')
plt.ylabel('Number of faster runs')
plt.subplot(122)
plt.semilogx(thresh, scores_it.T)
plt.legend(["NMF_LeeSeung",  "NeNMF_optimMajorant", "PGD", "NeNMF", "HALS"])
plt.title('How many times each algorithm reached threshold the fastest (iters)')
plt.xlabel('Rec error threshold')
plt.ylabel('Number of faster runs')

# Boxplots with errors after fixed time/iterations
xax = "noise_variance"
facet_row = "r"

fig_box = px.box(df, y="error_at_time_0.1", color="method", x=xax, facet_col=facet_row, log_y=True, template="plotly_white")
fig_box2 = px.box(df, y="error_at_time_0.5", color="method", x=xax, facet_col=facet_row, log_y=True, template="plotly_white")
fig_box3 = px.box(df, y="error_at_time_1", color="method", x=xax, facet_col=facet_row, log_y=True, template="plotly_white")
fig_box.update_xaxes(type='category')
fig_box2.update_xaxes(type='category')
fig_box3.update_xaxes(type='category')
fig_box.show()
fig_box2.show()
fig_box3.show()

fig_box_it = px.box(df, y="error_at_it_10", color="method", x=xax, facet_col=facet_row, log_y=True, template="plotly_white")
fig_box_it_2 = px.box(df, y="error_at_it_50", color="method", x=xax, facet_col=facet_row, log_y=True, template="plotly_white")
fig_box_it_3 = px.box(df, y="error_at_it_300", color="method", x=xax, facet_col=facet_row, log_y=True, template="plotly_white")
fig_box_it.update_xaxes(type='category')
fig_box_it_2.update_xaxes(type='category')
fig_box_it_3.update_xaxes(type='category')
fig_box_it.show()
fig_box_it_2.show()
fig_box_it_3.show()

# Convergence plots with all runs

# Plotting a few curves for all methods
nb_run_show=5
maxtime = 0.5

df2 = pd.DataFrame()
for idx_pd,i in enumerate(df["full_error"]):
    if df.iloc[idx_pd]["seed_idx"]<nb_run_show:
        its = np.linspace(0,len(i)-1,len(i))
        df2=pd.concat([df2, pd.DataFrame({
            "it":its,
            "time": df.iloc[idx_pd]["full_time"],
            "rec_err":i,
            "method":df.iloc[idx_pd]["method"],
            "run":df.iloc[idx_pd]["seed_idx"],
            "inner": df.iloc[idx_pd]["NbIter_inner"],
            "m": df.iloc[idx_pd]["m"],
            "n": df.iloc[idx_pd]["n"],
            "r": df.iloc[idx_pd]["r"],
            "sigma": df.iloc[idx_pd]["noise_variance"]
        })], ignore_index=True)

# cutting time for more regular plots
df2 = df2[df2["time"]<maxtime]

# small preprocessing for grouping plots
df2["groups"] = list(zip(df2["r"],df2["n"],df2["m"],df2["sigma"]))

pxfig = px.line(df2, line_group="groups", x="time", y= "rec_err", color='method',facet_col="run",facet_row="inner",
              log_y=True,
              height=1000)
pxfig.update_layout(font = dict(size = 20))
pxfig2 = px.line(df2, line_group="groups", x="it", y= "rec_err", color='method',facet_col="run",facet_row="inner",
              log_y=True,
              height=1000)
pxfig2.update_layout(font = dict(size = 20))

pxfig.show()
pxfig2.show()