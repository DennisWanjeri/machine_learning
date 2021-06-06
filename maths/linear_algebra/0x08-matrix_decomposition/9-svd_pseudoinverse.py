#!/usr/bin/python3
"""pseudoinverse via svd"""
import numpy as np
from numpy.linalg import svd


A = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
    [0.7, 0.8]
])

print(A)
U, s, V = svd(A)
#reciprocals of s
d = 1.0/s
#create an nxn D matrix
D = np.zeros(A.shape)
#populate D ith nxn diagonal matrix
D[:A.shape[1], :A.shape[1]] = np.diag(d)
#calculate pseudoinverse
B = V.T.dot(D.T).dot(U.T)
print(B)
