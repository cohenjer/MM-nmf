from cmath import inf
import numpy as np

def find_best_at_all_thresh(df, thresh, batch_size):
    """
    This utility function find the method was the fastest to reach a given threshold, at each threshold in the list thres.

    Parameters:
    ----------
    df : Pandas DataFrame
         The dataframe containing the errors and timings for each algorithm at each iterations, for several runs.
         For details on the expected names, check synthethic_comparisons_Frobenius.py
         Because I am a lazy coder:
            - Batch size must be constant
            - The algorithms must always be storred in df in the same order

    thresh: list
            A list of thresholds to be used for computing which method was faster.

    batch_size: int
            Number of algorithm runs to compare for the max pooling. Should be a multiple (typically 1x) of the number of algorithms.

    Returns:
    -------
    scores_time: nd array
            A table "method x thresh" with how many times each method was the fastest to reach a given threshold. Here faster is understood in runtime.

    scores_it: nd array
            A table "method x thresh" with how many times each method was the fastest to reach a given threshold. Here faster is understood in number of iterations.
    """

    timings = []
    iterations = []
    # Strategy: we sweep each error and find at which time each threshold was attained
    for row_errs,errs in enumerate(df["full_error"]):
        pos = 0
        time_reached = []
        it_reached = []
        for pos_err,err in enumerate(errs):
            while pos<len(thresh) and err<thresh[pos]:
                # just in case several thresholds are beaten in one iteration
                time_reached.append(df["full_time"][row_errs][pos_err])
                it_reached.append(pos_err)
                pos+=1
        if len(time_reached)<len(thresh):
            time_reached = time_reached +( (len(thresh)-len(time_reached))*[np.Inf] )
            it_reached = it_reached +( (len(thresh)-len(it_reached))*[np.Inf] )
        timings.append(time_reached)
        iterations.append(it_reached)
    # casting as a numpy array (matrix) for slicing vertically
    timings = np.array(timings)
    iterations = np.array(iterations)

    # Then we find who is the winner for each batch and for each threshold
    Nb_batch = int(len(timings)/batch_size)  # should be integer without recast
    # reshaping timings into a 3-way tensor for broadcasting numpy argmax
    timings = np.reshape(timings, [Nb_batch,batch_size,len(thresh)])
    iterations = np.reshape(iterations, [Nb_batch,batch_size,len(thresh)])
    # we can now find which count how many times each algorithm was faster by finding the index of the fastest method for each batch
    winners_time = my_argmin(timings)
    winners_it = my_argmin(iterations)
    # Assuming results are stored always in the same order, a batch row index corresponds to an algorithm name
    scores_time = np.zeros((batch_size,len(thresh)))
    scores_it = np.zeros((batch_size,len(thresh)))
    for k in range(batch_size):
        for i in range(Nb_batch): 
            for j in range(len(thresh)):
                if type(winners_time[i,j])==list:
                    if k in winners_time[i,j]:
                        scores_time[k,j]+=1
                else:
                    if winners_time[i,j]==k:
                        scores_time[k,j]+=1
                    
                if type(winners_it[i,j])==list:
                    if k in winners_it[i,j]:
                        scores_it[k,j]+=1
                else:
                    if winners_it[i,j]==k:
                        scores_it[k,j]+=1

    return scores_time, scores_it, timings, iterations

def my_argmin(a):
    """
    argmin but returns list of equal indices. Axis must be 1, a is a third order tensor.
    """
    tutu = a.min(axis=1)[:,None]
    tutu[tutu==np.Inf]=0 #removing np.Inf
    minpos = (a == tutu)
    # TODO: remove np.Inf counting
    npargmin = np.argmin(a,axis=1)
    myargmin= np.zeros(npargmin.shape, dtype=object)-1
    for i in range(minpos.shape[0]):
        for j in range(minpos.shape[1]):
            for k in range(minpos.shape[2]):
                if minpos[i,j,k]:
                    if type(myargmin[i,k])==list:
                        myargmin[i,k] = myargmin[i,k] + [j]
                    elif myargmin[i,k]==-1:
                        myargmin[i,k] = j
                    else:
                        myargmin[i,k] = [myargmin[i,k]] + [j]

    return myargmin