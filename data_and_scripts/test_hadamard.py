import numpy as np
from scipy.linalg import hadamard
import matplotlib.pyplot as plt

'''
 The following example build an Hadamard matrix A, which is used to measure an image x which is sparse (bitmap image). The observations are
    y = A_S x + noise
 where the noise level is user-set. The A_S matrix is a composed of a subset of rows or columns of A to make the least-squares problem ill-posed or strongly convex.
 The problem parameters are:
 m : int, n=2^m is the size of A (square matrix)
 p : int, number of rows (if p>0) or columns (if p<0) removed from A to A_S
 l : int, number of images (always 1 for the first part dealing with nnls)
 posA : bool, choose if A is nonnegative or not (in that case, the negative values are split into new rows, multiplying the number of rows by 2)
'''

#-------------------------------------------------------------------------
# Modeling

# hyperparameters (NNLS suggestion)
# posA = False
#m = 10
#n = 2**m
#p = 200
#if p<0:
    #n_col = 2**m+p
    #if posA:
    #    n_row = 2*n
    #else:
    #    n_row = n
#else:
    #n_col = n
    #n_row = 2**m-p
#l = 50
#sqrt_n_col = int(np.sqrt(n_col))

# (NMF suggestions)
posA = True
m = 10
n = 2**m
p = -1008#-999 # X images are then 4x4 # 5x5
if p<0:
    n_col = 2**m+p
    if posA:
        n_row = 2*n
    else:
        n_row = n
else:
    n_col = n
    n_row = 2**m-p
l = 5000
sqrt_n_col = int(np.sqrt(n_col))

# sensing operator is Hadamard
A = hadamard(n).astype('float')
# Process A if nonnegative A is required
if posA:
    # splitting negative and positive values in two rows
    Aneg = -np.minimum(A,0)
    Apos = np.maximum(A,0)
    A = np.zeros([2*n,n])
    A[:int(n_row/2),:] = Apos
    A[int(n_row/2):,:] = Aneg
# if p negative, remove columns, otherwise remove rows
A = A[:n_row,:n_col]

# image generation
X = np.random.randn(n_col,l) # l images
X[X>0] = 1
X[X<0] = 0
x = X[:,0] # one image

# Generatin observations
sig = 0.1
noise = np.random.randn(n_row,l)
Y = A@X + sig*noise
y = Y[:,0]

#-----------------------------------------------------------------
# Solving with nonnegative least squares
from tensorly.tenalg.proximal import fista

# Nonnegative Least Squares
out = fista(A.T@y, A.T@A, tol=1e-16, n_iter_max=1000)
# postprocessing into bitmap
out_bit = np.copy(out)
out_bit[out_bit<0.5]=0
out_bit[out_bit>=0.5]=1

# numerical results
err_fista = np.linalg.norm(x - out)/np.linalg.norm(x)
err_fista_bit = np.linalg.norm(x - out_bit)/np.linalg.norm(x)
print('Relative error of FISTA is {}'.format(err_fista))
print('Relative error of FISTA after bitmap projection is {}'.format(err_fista_bit))

# visual results
plt.figure()
plt.subplot(131)
plt.imshow(np.reshape(out,(sqrt_n_col,sqrt_n_col)))
plt.title('FISTA nnls solver')
plt.subplot(132)
plt.imshow(np.reshape(out_bit,(sqrt_n_col,sqrt_n_col)))
plt.title('FISTA nnls solver bitmap')
plt.subplot(133)
plt.imshow(np.reshape(x,(sqrt_n_col,sqrt_n_col)))
plt.title('True solution')
#plt.show()

#---------------------------------------------------------------------------
# NMF with several measurements
#from tensorly.decomposition._nn_cp import non_negative_parafac_hals as nmf_hals
from nn_fac.nmf import nmf as nmf_hals
from scipy.optimize import linear_sum_assignment as munkres

# NMF with HALS (beta=2)
W, H, err_nmf,_ = nmf_hals(Y, n_col, return_costs=True)
# postprocessing
# Permutation ambiguityH_bit = H_bit / np.sum(H_bit,axis=0) # scale before conversion
Ht = H.T
H_norm = Ht / np.sum(Ht, axis=0)
X_norm = X / np.sum(X, axis=0)
_, perms = munkres(-X_norm@H_norm)
W = W[:,perms]
H = H[perms,:]
# Bitmap conversion
H_bit = np.copy(H)
mean_H = np.mean(H_bit, axis=0)
H_bit[H_bit<mean_H]=0
H_bit[H_bit>=mean_H]=1

# Final relative error
errf_nmf = np.linalg.norm(Y - W@H, 'fro')/np.linalg.norm(Y, 'fro')
print('Relative error of NMF is {}'.format(errf_nmf))

# Printing a few images
plt.figure()
for i in range(5):
    plt.subplot(3,5,i+1)
    plt.imshow(np.reshape(H[:,i],(sqrt_n_col,sqrt_n_col)))
    plt.title('Estimated image {}'.format(i))
    plt.subplot(3,5,i+6)
    plt.imshow(np.reshape(H_bit[:,i],(sqrt_n_col,sqrt_n_col)))
    plt.title('Estimated bitmap {}'.format(i))
    plt.subplot(3,5,i+11)
    plt.imshow(np.reshape(X[:,i],(sqrt_n_col,sqrt_n_col)))
    plt.title('True bitmap {}'.format(i))
