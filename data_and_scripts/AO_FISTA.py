import numpy as np
import time

def ao_fg(Y, rank, step_rel=1, n_iter_max=100, init='random', tol=1e-8, verbose=0, inner_iter_max = 40, tol_inner = 1e-3):
    """NMF solved via (approximate) alternating optimization, with Nesterov's fast gradient as subsolver (sometimes called NeNMF)

        minimize 1/2 \|Y -AB.T\|_F^2 wrt A>=0, B>=0
        minimize 1/2 \|Y.T -BA.T\|_F^2 wrt A>=0, B>=0

    Inner loop stops after a maximum number of iterations is reached, or when the ratio of the current gradient norm on the initial inner loop gradient is small.

    TODO: see why inner stopping criterion is bugged
    check details in paper for stopping

    Parameters
    ----------
    Y : ndarray
    rank  : int
        Number of components.
    step_rel : float in [0,1]. Default: 1
        Percentage of theoretical largest stepsize actually used. Set smaller than 1 if divergence occurs.
    n_iter_max : int
        Maximum number of iteration
    init : {'random', list}, optional
        Type of factor matrix initialization.
        If a list is passed, this is directly used for initalization of A and B.
    tol : float, optional
        (Default: 1e-8) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    verbose : int, optional
        Level of verbosity
    inner_iter_max : int,
        maximum number of inner iterations for the Fast Gradient
    tol_inner : float,
        (Default: 1e-4) Tolerance for the inner stopping criterion

    Returns
    -------
    A : numpy array,
        Estimated templates

    B : numpy array,
        Estimated activations

    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
    Guan, N., Tao, D., Luo, Z., & Yuan, B. (2012). NeNMF: An optimal gradient method for nonnegative matrix factorization. IEEE Transactions on Signal Processing, 60(6), 2882-2898.
    """

    # only need to initialize B
    if init=='random':
        B = np.random.randn(Y.shape[1],rank)
        A = np.random.randn(Y.shape[0],rank)
    else:
        A = np.copy(init[0])
        B = np.copy(init[1])

    Ynorm = np.linalg.norm(Y)**2
    rec_errors = [np.linalg.norm(Y - A@B.T)**2]
    time_its = []

    # Initial gradients for inner stopping criterion
    grad_A = A@B.T@B - Y@B
    np.where(A>0, grad_A, np.minimum(grad_A,0))
    grad_B = B@A.T@A - Y.T@A
    np.where(B>0, grad_B, np.minimum(grad_B,0))
    grad_norm_zero = np.linalg.norm(np.concatenate((grad_A,grad_B))) # a little costly

    for iteration in range(n_iter_max):

        # Tracking time
        time0 = time.time()

        if verbose > 1:
            print("Starting iteration", iteration + 1)

        # A update
        # Precomputations
        if iteration==0 or not (tol or return_errors):
            BtB = B.T@B
        YB = Y@B
        # stepsize inverse of Lipschitz constant
        step = step_rel/np.linalg.norm(BtB,2)
        # initialize extrapolation
        Z = np.copy(A)
        beta = 0
        for inner_iteration in range(inner_iter_max):
            # one iteration of Nesterov Fast Gradient
            # Gradient step
            A_old = np.copy(A)
            A = np.maximum(Z - step *(Z@BtB - YB), 0)
            # KKT conditions for early stopping
            grad_A = A@BtB - YB
            np.where(A>0, grad_A, np.minimum(grad_A,0))
            grad_A_norm = np.linalg.norm(grad_A) # a little costly
            # Extrapolation
            beta_old = beta
            beta = (1+np.sqrt(4*beta_old**2+1))/2
            Z = A + (beta_old-1)/beta*(A - A_old)
            if (grad_A_norm/grad_norm_zero)<tol_inner:
                break

        if verbose:
            print('number of inner iterations for A: {}'.format(inner_iteration))

        # A update
        # Precomputations
        AtA = A.T@A
        YtA = Y.T@A
        # stepsize inverse of Lipschitz constant
        step = step_rel/np.linalg.norm(AtA,2)
        # initialize extrapolation
        Z = np.copy(B)
        beta = 0
        for inner_iteration in range(inner_iter_max):
            # one iteration of Nesterov Fast Gradient
            # Gradient step
            B_old = np.copy(B)
            B = np.maximum(Z - step *(Z@AtA - YtA), 0)
            # KKT conditions for early stopping
            grad_B = B@AtA - YtA
            np.where(B>0, grad_B, np.minimum(grad_B,0))
            grad_B_norm = np.linalg.norm(grad_B) # a little costly
            # Extrapolation
            beta_old = beta
            beta = (1+np.sqrt(4*beta_old**2+1))/2
            Z = B + (beta_old-1)/beta*(B - B_old)
            if (grad_B_norm/grad_norm_zero)<tol_inner:
                break

        if verbose:
            print('number of inner iterations for B: {}'.format(inner_iteration))


        # Calculate the current unnormalized error if we need it
        if (tol or return_errors):
            #err = np.linalg.norm(Y - DX@B.T)
            # faster version
            # normY + <AB.T,AB.T> - 2<Y.T,BA.T>
            BtB = B.T@B  # keep it for next iteration
            err = Ynorm + np.sum(AtA*BtB) - 2*np.sum(YtA*B)

        rec_errors.append(err)


        if iteration >= 1:
            rel_rec_error_decrease = np.abs(rec_errors[-2] - rec_errors[-1])/rec_errors[-2]
            if verbose:
                print("iteration {}, reconstruction error: {}, relative decrease = {}".format(iteration, err, rel_rec_error_decrease))
            if rel_rec_error_decrease < tol:
                if verbose:
                    print("NMF converged after {} iterations".format(iteration))
                break

        else:
            if verbose:
                print('reconstruction error={}'.format(rec_errors[-1]))

        # Storing time
        time_its.append(time.time() - time0)

    return [A, B], rec_errors, time_its
