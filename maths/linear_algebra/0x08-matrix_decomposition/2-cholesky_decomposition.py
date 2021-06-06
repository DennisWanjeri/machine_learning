#!/usr/bin/python3
"""Cholesky decomposition
is for square symmetric matrices where all values are greater than zero, positive definite matrices"""
import numpy as np
from numpy.linalg import cholesky


#define symmetrical matrix
A = np.array([
    [2, 1, 1],
    [1, 2, 1],
    [1, 1, 2]
])
print(A)
#factorize
L = cholesky(A)
print(L)
#reconstruct
B = L.dot(L.T)
print(B)
