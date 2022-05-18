import numpy as np
from matplotlib import pyplot as plt
import NMF_Frobenius as nmf_f 
import time
import nn_fac
import pandas as pd
import plotly.express as px
import utils

plt.close('all')
# Fixe the matrix sizes

# Seeding
#np.random.seed(hash(´toto'))

# Storing intermediate results
df = pd.DataFrame()

# --------------------- Choose parameters for grid tests ------------ #
# dimensions
m_list = [20]
n_list = [20]
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
sigma_list = [0,1e-7]

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
                        error0, W0, H0, toc0 = nmf_f.NMF_Lee_Seung(V,  Wini, Hini, NbIter, NbIter_inner,tol=tol)
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

                        df = pd.concat([df,pd.DataFrame(
                            {
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
                                "error_at_time_thr": [err_time_0,err_time_1,err_time_2,err_time_3,err_time_4],
                                "error_at_at_thr": [err_it_0,err_it_1,err_it_2,err_it_3,err_it_4],
                                "final_error": [error0[-1], error1[-1], error2[-1], error3[-1], error4[-1]],
                                "total_time": [toc0[-1], toc1[-1], toc2[-1], toc3[-1], toc4[-1]],
                                "full_error": [error0,error1,error2,error3,error4],
                                "full_time": [toc0,toc1,toc2,toc3,toc4],
                                "NbIterStop": [len(error0),len(error1),len(error2),len(error3),len(error4)]
                            }
                        )], ignore_index=True)


# testing post-processing
thresh = np.logspace(-3,-8,50) 
scores_time, scores_it, timings, iterations = utils.find_best_at_all_thresh(df,thresh, 5)

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



#fig = plt.figure(figsize=(6,3),tight_layout = {'pad': 0})

#plt.semilogy(error0, label = 'Lee and Seung', linewidth = 3)
#plt.semilogy(error1,'--', label = 'Pham et al', linewidth = 3)
#plt.semilogy(error2,'--', label = 'Gradient descent', linewidth = 3)   
#plt.semilogy(error3,'--', label = 'NeNMF', linewidth = 3)
#plt.semilogy(error4,'--', label = 'HALS', linewidth = 3)
#plt.title('Objective function values versus iterations', fontsize=14)# for different majorizing functions')
#plt.xlabel('Iteration', fontsize=14)
#plt.ylabel(r'$\log\left( || V - WH ||/nm \right)$', fontsize=14)
#plt.legend(fontsize = 14)
#plt.grid(True)

#fig2 = plt.figure(figsize=(6,3),tight_layout = {'pad': 0})

#plt.semilogy(toc0, error0, label = 'Lee and Seung', linewidth = 3)
#plt.semilogy(toc1, error1,'--', label = 'Pham et al', linewidth = 3)
#plt.semilogy(toc2, error2,'--', label = 'Gradient descent', linewidth = 3)   
#plt.semilogy(toc3, error3,'--', label = 'NeNMF', linewidth = 3)
#plt.semilogy(toc4, error4,'--', label = 'HALS', linewidth = 3)
#plt.title('Objective function values versus time', fontsize=14)# for different majorizing functions')
#plt.xlabel('Time (s)', fontsize=14)
#plt.ylabel(r'$\log\left( || V - WH ||/nm \right)$', fontsize=14)
#plt.legend(fontsize = 14)
#plt.grid(True)


#print('Lee and Seung: Error = '+str(np.mean(Error0)) + '; NbIter = '  + str(np.mean(NbIterStop0)) + '; Elapsed time = '+str(time0)+ '\n')
#print('Pham et al: Error = '+str(np.mean(Error1)) + '; NbIter = '  + str(np.mean(NbIterStop1)) + '; Elapsed time = '+str(time1)+ '\n')
#print('Gradient descent: Error = '+str(np.mean(Error2)) + '; NbIter = '  + str(np.mean(NbIterStop2)) + '; Elapsed time = '+str(time2)+ '\n')
#print('NeNMF: Error = '+str(np.mean(Error3)) + '; NbIter = '  + str(np.mean(NbIterStop3)) + '; Elapsed time = '+str(time3)+ '\n')
#print('HALS: Error = '+str(np.mean(Error4)) + '; NbIter = '  + str(np.mean(NbIterStop4)) + '; Elapsed time = '+str(time4)+ '\n')

# Plots with all runs

# Plotting a few curves for all methods
nb_run_show=5

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
            "inner": df.iloc[idx_pd]["NbIter_inner"]
        })], ignore_index=True)

# cutting time for more regular plots
maxtime = 0.5

df2 = df2[df2["time"]<maxtime]

pxfig = px.line(df2, x="time", y= "rec_err",color='method',facet_col="run",facet_row="inner",
              log_y=True,
              height=1000)
pxfig.update_layout(font = dict(size = 20))
pxfig2 = px.line(df2, x="it", y= "rec_err",color='method',facet_col="run",facet_row="inner",
              log_y=True,
              height=1000)
pxfig2.update_layout(font = dict(size = 20))

pxfig.show()
pxfig2.show()