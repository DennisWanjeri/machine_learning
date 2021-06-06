#!/usr/bin/python3
"""LU decomposition"""
import numpy as np
from scipy.linalg import lu


A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(A)
#factorize
P, L, U = lu(A)
print(P)
print(L)
print(U)
#reconstruct
B = P.dot(L).dot(U)
print(B)
