#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 22:14:39 2022

@author: pham
"""


import logging
import time

from numpy.random import RandomState
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
 
import numpy as np
from glob import glob
from skimage.io import imread
import os

from NMF_KL import Proposed_KL, Fevotte_KL

def faceRecognition(img, V, W, H):
    ri, ci = img.shape
    img = img.reshape(-1)
    
    WtW = W.T.dot(W)
    if np.linalg.det(WtW) == 0:
        Ht = np.linalg.pinv(WtW).dot(W.T).dot(img)
    else:
        Ht = np.linalg.inv(WtW).dot(W.T).dot(img)
        
    s = np.zeros(H.shape[1])
    for i in range(H.shape[1]):
        s[i]=(1/(np.linalg.norm(H[:,i]) *np.linalg.norm(Ht)))*H[:,i].dot(Ht)
    
    h_threshold = 0.86
    if max(s)>=h_threshold:
        i= np.argmax(s)
        return V[:,i].reshape(ri,ci)
    else:
        return []
    


if __name__ == '__main__':

 
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    n_row, n_col = 2, 3
    n_components = n_row * n_col
    image_shape = (64, 64)
    rng = RandomState(0)
    
    # #############################################################################
    # Load faces data
    dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
    # pick randomtly an image
    name_img = np.random.randint(0, 400)
    
    img = dataset.data[name_img,:].reshape(image_shape) 
    V = np.delete(dataset.data, name_img, axis=0).T
    V[V<=0] = 1e-8
    
    
    # Init for NFM
    
    ind0=None
    ind1=None
    nb_inner=10
    NbIter=1000
    epsilon=1e-8
    tol=(1e-6) *V.size
    r = 13
    Wini = np.random.rand(V.shape[0],r)
    Hini = np.random.rand(r, V.shape[1])
    
    #--- Fevotte 
    time_start0 = time.time()  

    crit0, W0, H0 = Fevotte_KL(V,  Wini, Hini, ind0, ind1, nb_inner, NbIter, epsilon, tol)
    crit0 = np.array(crit0)/V.size
    
    I0 = faceRecognition(img, V, W0, H0)
    time0 = time.time()-time_start0
    
    # -- proposed
    
    time_start1 = time.time()  
    sumH = 0.1
    sumW = 0.1
    crit1, W1, H1 = Proposed_KL(V, Wini, Hini, sumH, sumW,  ind0, ind1, nb_inner, NbIter, epsilon, tol)
    crit1 = np.array(crit1)/V.size
    I1 = faceRecognition(img, V, W1, H1)
    time1 = time.time()-time_start1
    
    #%% ------------------Display objective functions
    cst = 1e-12 
    
    print('Fevotte et al: Crit = ' + str(crit0[-1]) + '; NbIter = '  + str(len(crit0)) + '; Elapsed time = '+str(time0)+ '\n')
    print('Pham et al: Crit = ' + str(crit1[-1]) + '; NbIter = '  + str(len(crit1)) + '; Elapsed time = '+str(time1)+ '\n')

     
    plt.figure(figsize=(6,3),tight_layout = {'pad': 0})    
    plt.semilogy(crit0 + cst, label = 'Fevotte et al', linewidth = 3)
    plt.semilogy(crit1 + cst, label = 'Pham et al', linewidth = 3)
    plt.title('Objective function values versus iterations', fontsize=14)# for different majorizing functions')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(r'$\log\left( || V - WH || \right)$', fontsize=14)
    plt.legend(fontsize = 14)
    plt.grid(True)
    
    
    if I0.all:
              
        fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10,5))
 
        
        ax0.imshow(img, cmap ='gray')
        ax0.set_title('Searching for')
        ax1.imshow(I0, cmap = 'gray')
        ax1.set_title('Fevotte et al found')
        
    else:    
        print('Fevotte et al: Algorithm does not find a face')
        
        
    if I1.all:
         
        fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10,5))
        ax0.imshow(img, cmap ='gray')
        ax0.set_title('Searching for')
        ax1.imshow(I1, cmap = 'gray')
        ax1.set_title('Pham et al found')
        
    else:    
        print('Pham et al: Algorithm does not find a face')
       
        
    
    
         
        
    
  
    


   
    
    
    
    
    