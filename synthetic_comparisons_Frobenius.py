import numpy as np
from matplotlib import pyplot as plt
import NMF_Frobenius as nmf_f 
import time
import nn_fac

# todo: 
# -track time
# -improve error comp

plt.close('all')
# Fixe the matrix sizes

# Seeding
#np.random.seed(hash(´toto'))

# dimensions
m = 500
n = 40
r = 5

# Max number of iterations
NbIter = 10000
# Fixed number of inner iterations
NbIter_inner= 10
# Stopping criterion error<tol
tol = 1e-5

# Fixed the signal 
Worig = np.random.rand(m, r) 
Horig = np.random.rand(r, n)  
Vorig = Worig.dot(Horig)

# noise variance
sigma = 0

# Number of random inits
NbSeed=1

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


# One noise, several inits; NMF is not unique and nncvx so we will find several results
for  s in range(NbSeed): #[NbSeed-1]:#
        
    # Initialization for H0 as a random matrix
    Hini = np.random.rand(r, n)
    Wini = np.random.rand(m, r) #sparse.random(rV, cW, density=0.25).toarray() 
    
    # adding noise to the observed data
    np.random.seed(s)
    N = sigma*np.random.rand(m,n)
    V = Vorig + N
    
    
    time_start0 = time.time()
    error0, W0, H0 = nmf_f.NMF_Lee_Seung(V,  Wini, Hini, NbIter, NbIter_inner,tol=tol)
    time0 = time.time() - time_start0
    Error0[s] = error0[-1] 
    NbIterStop0[s] = len(error0)
    
    
    
    time_start1 = time.time()
    error1, W1, H1  = nmf_f.NeNMF_optimMajo(V, Wini, Hini, tol=tol, itermax=NbIter, nb_inner=NbIter_inner)
    #error1, W1, H1 = NMF_proposed_Frobenius(V, Wini, Hini, NbIter, NbIter_inner, tol=tol)
    time1 = time.time() - time_start1
    Error1[s] = error1[-1] 
    NbIterStop1[s] = len(error1)
        
    time_start2 = time.time()
    error2, W2, H2  = nmf_f.Grad_descent(V , Wini, Hini, NbIter, NbIter_inner, tol=tol)
    time2 = time.time() - time_start1
    Error2[s] = error2[-1] 
    NbIterStop2[s] = len(error2)
    
    
    time_start3 = time.time()
    error3, W3, H3  = nmf_f.NeNMF(V, Wini, Hini, tol=tol, nb_inner=NbIter_inner, itermax=NbIter)
    time3 = time.time() - time_start3
    Error3[s] = error3[-1]
    NbIterStop3[s] = len(error3)

    time_start4 = time.time()
    # Stopping criterion in nnfac is difference of loss; I should hack it to use just loss
    # Also add first error to errors
    # And remove auto-acc
    W4, H4, error4,_ = nn_fac.nmf.nmf(V, r, init="custom", U_0=np.copy(Wini), V_0=np.copy(Hini), n_iter_max=NbIter, tol=tol, update_rule='hals',beta=2, return_costs=True, NbIter_inner=NbIter_inner)
    time4 = time.time() - time_start4
    Error4[s] = error4[-1]
    NbIterStop4[s] = len(error4)


fig = plt.figure(figsize=(6,3),tight_layout = {'pad': 0})

plt.semilogy(error0, label = 'Lee and Seung', linewidth = 3)
plt.semilogy(error1,'--', label = 'Pham et al', linewidth = 3)
plt.semilogy(error2,'--', label = 'Gradient descent', linewidth = 3)   
plt.semilogy(error3,'--', label = 'NeNMF', linewidth = 3)
plt.semilogy(error4,'--', label = 'HALS', linewidth = 3)
plt.title('Objective function values versus iterations', fontsize=14)# for different majorizing functions')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel(r'$\log\left( || V - WH || \right)$', fontsize=14)
plt.legend(fontsize = 14)
plt.grid(True)



print('Lee and Seung: Error = '+str(np.mean(Error0)) + '; NbIter = '  + str(np.mean(NbIterStop0)) + '; Elapsed time = '+str(time0)+ '\n')
print('Pham et al: Error = '+str(np.mean(Error1)) + '; NbIter = '  + str(np.mean(NbIterStop1)) + '; Elapsed time = '+str(time1)+ '\n')
print('Gradient descent: Error = '+str(np.mean(Error2)) + '; NbIter = '  + str(np.mean(NbIterStop2)) + '; Elapsed time = '+str(time2)+ '\n')
print('NeNMF: Error = '+str(np.mean(Error3)) + '; NbIter = '  + str(np.mean(NbIterStop3)) + '; Elapsed time = '+str(time3)+ '\n')
print('HALS: Error = '+str(np.mean(Error4)) + '; NbIter = '  + str(np.mean(NbIterStop4)) + '; Elapsed time = '+str(time4)+ '\n')