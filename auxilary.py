#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 14:11:31 2023

@author: pham
"""
import numpy as np

def auxilary(a,b,alpha,epsilon):
    ba = b/a 
    ind = np.argsort(ba)
    ba = ba[ind]
    a = a[ind]
    b = b[ind]
    
    mu = (np.cumsum(np.sqrt(a[::-1])*np.sqrt(b[::-1]))[::-1])**2 /((alpha - np.concatenate(([0],epsilon*np.cumsum(a[:-1]))))**2)
    u = epsilon*np.ones(a.shape) 
    ind_ = np.where(mu< (ba/epsilon**2) )[0]
    
        
    u[ind_[0]:] = np.sqrt(ba[ind_[0]:]/mu[ind_[0]])
     
    return u[np.argsort(ind)]
    
    

n = 100
a = np.random.rand(n)
b = np.random.rand(n)
epsilon = np.random.rand(1)[0]

 
 
alpha = 1

while epsilon*np.sum(a) > alpha:
    epsilon = epsilon/2

print('alpha - epsilon*a = ' + str(epsilon*np.sum(a)))


u = auxilary(a,b,alpha,epsilon)


print('sum(b/u) = ' + str(np.sum(b/u)) )
print('<a,u> = ' +str(a.dot(u)) )
print('min(u) = ' +str(np.min(u)))
print('epsilon = ' + str(epsilon))
