import numpy as np

def sparsify(M, s=0.5, epsilon=1e-8):
    """Adds zeroes in matrix M in order to have a ratio s of nnzeroes/nnentries.

    Parameters
    ----------
    M : 2darray
        The input numpy array
    s : float, optional
        the sparsity ratio (0 for fully sparse, 1 for density of the original array), by default 0.5
    """    
    vecM = M.flatten()
    # use quantiles
    val = np.quantile(vecM, 1-s)
    # put zeros in M
    M[M<val]=epsilon
    return M

def nearest_neighbour_H(Y, W, epsilon=0):
    
    # Normalization
    Ynorm = np.linalg.norm(Y, axis=0)
    Wnorm = np.linalg.norm(W, axis=0)
    Yn = Y/Ynorm
    Wn = W/Wnorm
    
    # Nearest neighbour
    angles = Wn.T@Yn
    nearest_n = np.argmax(angles, axis=0)
    
    # Updating H
    H = epsilon*np.ones([W.shape[1],Y.shape[1]])
    for i in range(H.shape[1]):
        H[nearest_n[i], i] = Y[:, i].sum()/W[:, nearest_n[i]].sum()
    
    return H

def absls(Y, W, epsilon=0):
    
    H = np.linalg.lstsq(W, Y)[0]
    #H[H < epsilon] = epsilon
    H = np.abs(H)
    H[H<epsilon] = epsilon
    
    return H

def opt_scaling(Y, WH):
    # columnwise for H
    
    return np.array([Y[:, i].sum()/WH[:, i].sum() for i in range(WH.shape[1])])