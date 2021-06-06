#!/usr/bin/python3
"""Confirm an Eigenvector and Eigenvalue
By multiplying the candidate eigenvector by the value vector and comparing the result with eigenvalue"""
import numpy as np
from numpy.linalg import eig


A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

#factorize
values, vectors = eig(A)
#confirm first eigenvector
B = A.dot(vectors[:, 0])
print(B)
C = vectors[:, 0] * values[0]
print(C)
