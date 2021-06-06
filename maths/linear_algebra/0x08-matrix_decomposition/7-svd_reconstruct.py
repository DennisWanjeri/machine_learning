#!/usr/bin/python3
"""Reconstruvting a matrix from svd elements"""
import numpy as np
from scipy.linalg import svd
from numpy import diag


A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
print(A)
U, s, V = svd(A)
#create nxn sigma matrix
Sigma = np.zeros((A.shape[0], A.shape[1]))
#populate Sigma with nxn diagonal matrix
Sigma[:A.shape[1], :A.shape[1]] = diag(s)
#reconstruct matrix
B = U.dot(Sigma.dot(V))
print(B)
