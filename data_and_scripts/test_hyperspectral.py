import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import time
# this is the package that my PhD student Axel has made. It has an good implementation of HALS (Frobenius norm).
# pip install nn-fac
import nn_fac
# mm code
#from NMF_Frobenius import NMF_proposed_Frobenius

'''
    We load an hyperspectral image called Urban. It has 162 clean spectral bands, and 307x307 pixels. We also load a set of good endmembers considered as ``Ground Truth'' (rank=6 spectra), and we define subsets of the image that are likely to contain pue pixels.

    For the NNLS part, we can use either W as the ground truth (tall matrix) or candidates pixels (if more than 162, fat matrix), and estimate abundances H that should be very sparse.

    For the NMF, we estimate both W and H, and plot the results. In this experiment, the Frobenius norm is well adapted.
'''

#-------------------------------------------------------------------------
# Data import and preprocessing

# Loading the data
dict = scipy.io.loadmat('Urban.mat')

# dict is a python dictionnary. It contains the matrix we want to NMF
M = np.transpose(dict['A']) # permmtation because we like spectra in W
print(M.shape)

# It can be nice to normalize the data, then absolute error is also relative error
M = M/np.linalg.norm(M, 2)

# Ground truth import (todo)

# Candidate pixels extraction (todo)


# --------------------------------------------------------------------
# NMF computations

# Explicit init
rank = 5
W0 = np.random.rand(M.shape[0], rank)
H0 = np.random.rand(rank, M.shape[1])

# Now compute the NMF of M
n_iter_max = 300
t0 = time.time()
out = nn_fac.nmf.nmf(M, rank, n_iter_max = n_iter_max, tol = 1e-8, verbose = True, return_costs=True, U_0=W0, V_0=H0)
tHALS = time.time() - t0

# Comparing with Quyen's code
# To compare the two algorithms it is nice if they have about the same per-iteration complexity
# Is it almost free to increase inner_iter?
#inner_iter = 30 # HALS uses a dynamic stopping criterion, we could do the same
#t0 = time.time()
#out2 = NMF_proposed_Frobenius(M, M, W0, H0, n_iter_max, inner_iter) TODO
#tmm = time.time() - t0

#---------------------------------------------------
# Post-processing

print('HALS time per iteration:', tHALS/100)
#print('MM time per iteration:', tmm/n_iter_max)

# Reconstruct M to compute relative error
W = out[0]
H = out[1]
Mest = W@H

err = np.linalg.norm(M - Mest, 2)
print(err)

# Reconstruction with mm
#Wmm = out2[1]
#Hmm = out2[2]
#Mestmm = Wmm@Hmm
#
#errmm = np.linalg.norm(M - Mestmm, 2)
#print(errmm)

# Printing the convergence plot
plt.figure()
# no initial error in HALS error
xhals = np.linspace(0,tHALS,n_iter_max)
#xmm = np.linspace(0,tmm,n_iter_max)
plt.semilogy(xhals,out[2])
#plt.semilogy(xmm,out2[0][1:])
plt.xlabel('fake time (iter/avg iter time)')
plt.legend('HALS')

plt.show()

# Normalisations
W = W/np.sqrt(np.sum(W**2,axis=0))
#Wmm = Wmm/np.sqrt(np.sum(Wmm**2,axis=0))

# Visualization of the results
# TODO: PERMUTE COLUMNS SO THAT THEY MATCH
plt.subplot(3,5,2)
plt.plot(W)
plt.legend([1,2,3,4,5])
#plt.subplot(3,5,3)
#plt.plot(Wmm)
#plt.legend([1,2,3,4,5])
plt.subplot(356)
plt.imshow(np.transpose(np.reshape(H[0, :], [307,307])))
plt.title('1')
plt.subplot(357)
plt.imshow(np.transpose(np.reshape(H[1, :], [307,307])))
plt.title('2')
plt.subplot(358)
plt.imshow(np.transpose(np.reshape(H[2, :], [307,307])))
plt.title('3')
plt.subplot(359)
plt.imshow(np.transpose(np.reshape(H[3, :], [307,307])))
plt.title('4')
plt.subplot(3,5,10)
plt.imshow(np.transpose(np.reshape(H[4, :], [307,307])))
plt.title('5')
#plt.subplot(3,5,11)
#plt.imshow(np.transpose(np.reshape(Hmm[0, :], [307,307])))
#plt.title('1')
#plt.subplot(3,5,12)
#plt.imshow(np.transpose(np.reshape(Hmm[1, :], [307,307])))
#plt.title('2')
#plt.subplot(3,5,13)
#plt.imshow(np.transpose(np.reshape(Hmm[2, :], [307,307])))
#plt.title('3')
#plt.subplot(3,5,14)
#plt.imshow(np.transpose(np.reshape(Hmm[3, :], [307,307])))
#plt.title('4')
#plt.subplot(3,5,15)
#plt.imshow(np.transpose(np.reshape(Hmm[4, :], [307,307])))
#plt.title('5')
plt.show()
