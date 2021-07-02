import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import time

# this is the package that my PhD student Axel has made. It has an good implementation of HALS (Frobenius norm).
# pip install nn-fac
import nn_fac

# mm code
from NMF_Frobenius import NMF_proposed_Frobenius

dict = scipy.io.loadmat('Urban.mat')

# dict is a python dictionnary. It contains the matrix we want to NMF
M = np.transpose(dict['A']) # permmtation because we like spectra in W
print(M.shape)

# It can be nice to normalize the data, then absolute error is also relative error
M = M/np.linalg.norm(M, 2)

# Explicit init
rank = 5
W0 = np.random.rand(M.shape[0], rank)
H0 = np.random.rand(rank, M.shape[1])

# Now compute the NMF of M
t0 = time.time()
out = nn_fac.nmf.nmf(M, rank, init='random', n_iter_max = 100, tol = 1e-8, verbose = True, return_errors=True, U_0=W0, V_0=H0)
tHALS = time.time() - t0

# Timings

# Comparing with Quyen's code
n_iter_max = 100
# To compare the two algorithms it is nice if they have about the same per-iteration complexity
# Is it almost free to increase inner_iter?
inner_iter = 20 # HALS uses a dynamic stopping criterion, we could do the same
t0 = time.time()
out2 = NMF_proposed_Frobenius(M, M, W0, H0, n_iter_max, inner_iter)
tmm = time.time() - t0

print('HALS time per iteration:', tHALS/100)
print('MM time per iteration:', tmm/n_iter_max)

# Reconstruct M to compute relative error
W = out[0]
H = out[1]
Mest = W@H

err = np.linalg.norm(M - Mest, 2)
print(err)

# Reconstruction with mm
Wmm = out2[1]
Hmm = out2[2]
Mestmm = Wmm@Hmm

errmm = np.linalg.norm(M - Mestmm, 2)
print(errmm)

# Printing the convergence plot
plt.figure()
# no initial error in HALS error
xhals = np.linspace(0,tHALS,100)
xmm = np.linspace(0,tmm,101)
plt.semilogy(xhals,out[2])
plt.semilogy(xmm,out2[0])
plt.xlabel('fake time (iter/avg iter time)')
plt.legend(['HALS', 'MM'])

plt.show()

# Normalisations
W = W/np.sqrt(np.sum(W**2,axis=0))
#H = H/np.sqrt(np.sum(H**2,axis=1)) # bug
Wmm = Wmm/np.sqrt(np.sum(Wmm**2,axis=0))
#Hmm = Hmm/np.sqrt(np.sum(Hmm**2,axis=1))

# Visualization of the results
plt.subplot(3,5,2)
plt.plot(W)
plt.legend([1,2,3,4,5])
plt.subplot(3,5,3)
plt.plot(Wmm)
plt.legend([1,2,3,4,5])
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
plt.subplot(3,5,11)
plt.imshow(np.transpose(np.reshape(Hmm[0, :], [307,307])))
plt.title('1')
plt.subplot(3,5,12)
plt.imshow(np.transpose(np.reshape(Hmm[1, :], [307,307])))
plt.title('2')
plt.subplot(3,5,13)
plt.imshow(np.transpose(np.reshape(Hmm[2, :], [307,307])))
plt.title('3')
plt.subplot(3,5,14)
plt.imshow(np.transpose(np.reshape(Hmm[3, :], [307,307])))
plt.title('4')
plt.subplot(3,5,15)
plt.imshow(np.transpose(np.reshape(Hmm[4, :], [307,307])))
plt.title('5')
plt.show()
