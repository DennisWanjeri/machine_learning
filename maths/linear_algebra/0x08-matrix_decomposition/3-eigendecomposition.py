#!/usr/bin/python3
"""Eigendecomposition
involves decomposing a square matrix into a set of eigenvectors and eigenvalues"""
import numpy as np
from numpy.linalg import eig


A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(A)
values, vectors = eig(A)
print(values)
print(vectors)
