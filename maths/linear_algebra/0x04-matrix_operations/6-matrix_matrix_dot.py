#!/usr/bin/python3
"""matrix-matrix dot product"""
import numpy as np


A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

B = np.array([
    [1, 2],
    [3, 4]
])
#can use the dot method or @
C = A.dot(B)
print(A.shape)
print(B.shape)
print(C.shape)
print(C)
#@ operator
#D = A @ B

#print(D)
