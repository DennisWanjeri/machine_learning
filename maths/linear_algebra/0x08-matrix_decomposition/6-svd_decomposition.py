#!/usr/bin/python3
"""Calculating singular value decomposition"""
import numpy as np
from scipy.linalg import svd


A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
print(A)
#factorize
U, s, V = svd(A)
print(U)
print(s)
print(V)
