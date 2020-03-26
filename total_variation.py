# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:47:23 2020

@author: Jochem Mullink
"""


import cvxpy as cv
import numpy as np
from math import floor
        

if __name__ == '__main__':
    #Size the picture
    k,l = 100,100

    #Generate a test array A.
    A = np.zeros((k,l))
    A[floor(k/3):floor(k/2),floor(l/3):floor(l/2)] = 1
    
    lamb = 0.5
    
    u = cv.Variable(shape=(k,l))
    obj = cv.Minimize(cv.atoms.sum_squares(A-u) + lamb*cv.tv(u))
    prob = cv.Problem(obj)
    prob.solve(verbose=True, solver=cv.SCS,eps=1e-6)
    
    sol = u.value