#!/usr/bin/python3
"""QR Decomposition
is for n x m matrices and decomposes a matrix into Q and R components
A = Q.R"""
import numpy as np
from numpy.linalg import qr


A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
print(A)
#factorize
Q, R = qr(A, 'complete')
print(Q)
print(R)
#reconstruct
B = Q.dot(R)
print(B)
