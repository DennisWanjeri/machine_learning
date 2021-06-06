#!/usr/bin/python3
"""Reconstruct matrix
given the eigenvectors and eigenvalues"""
import numpy as np
from numpy.linalg import eig, inv


A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(A)
values, vectors = eig(A)
#create a matrix from eigenvectors
Q = vectors
#create inverse of eigenvectors matrix
R = inv(Q)
#create diagonal matrix from eigenvalues
L = np.diag(values)
#reconstruct original matrix
B = Q.dot(L).dot(R)
print(B)
