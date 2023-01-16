#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 14:11:31 2023

@author: pham
"""
import numpy as np

def auxilary(c,b,normv,epsilon):
    bc = b/c 
    ind = np.argsort(bc)
    bc = bc[ind]
    c = c[ind]
    b = b[ind]
    
    mu = (np.cumsum(np.sqrt(c[::-1])*np.sqrt(b[::-1]))[::-1])**2 /((normv - np.concatenate(([0],epsilon*np.cumsum(c[:-1]))))**2)
    u = epsilon*np.ones(c.shape) 
    ind_ = np.where(mu< (bc/epsilon**2) )[0]
    
        
    u[ind_[0]:] = np.sqrt(bc[ind_[0]:]/mu[ind_[0]])
     
    return u[np.argsort(ind)]
    
    

n = 100
c = np.random.rand(n)
b = np.random.rand(n)
epsilon = np.random.rand(1)[0]

 
 
normv = 1

while epsilon*np.sum(c) > normv:
    epsilon = epsilon/2

print('normv - epsilon*a = ' + str(epsilon*np.sum(c)))


u = auxilary(c,b,normv,epsilon)


print('sum(b/u) = ' + str(np.sum(b/u)) )
print('<a,u> = ' +str(c.dot(u)) )
print('min(u) = ' +str(np.min(u)))
print('epsilon = ' + str(epsilon))
