#!/usr/bin/python3
"""Dimensionality Reduction
Reducing data with a large number of features(columns) than observations(rows) most relevant"""
import numpy as np
from numpy import diag
from scipy.linalg import svd


A = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
])
print(A)
U, s, V = svd(A)
#create a sigma matrix
Sigma = np.zeros((A.shape[0], A.shape[1]))
#populate sigma with nx n diagonal matrix
Sigma[:A.shape[0], :A.shape[0]] = diag(s)
#select
n_elements = 2
Sigma = Sigma[:, :n_elements]
V =V[:n_elements, :]
#reconstruct
B = U.dot(Sigma.dot(V))
print(B)
#transform
T = U.dot(Sigma)
print(T)
T = A.dot(V.T)
print(T)
