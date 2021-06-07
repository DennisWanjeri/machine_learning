#!/usr/bin/python3
"""calculating Principal Component Analysis from scratch using Numpy"""
import numpy as np
from numpy import mean, cov
from numpy.linalg import eig


A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
print(A)
#get column means
M = mean(A.T, axis=1)
print(M)
#center column by subtracting column means
C = A - M
print(C)
#calculate covariance matrix of centred matrix
V = cov(C.T)
print(V)
#factorize covariance matrix
values, vectors = eig(V)
print("----------")
print(values)
print(vectors)
#project data
P = vectors.T.dot(C.T)
print(P.T)
