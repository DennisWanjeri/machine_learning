#!/usr/bin/python3
"""Orthogonal Matrix
two vectors are orthogonal when their dot product equals zero"""
import numpy as np
from numpy.linalg import inv


#define orthogonal matrix
Q = np.array([
    [1, 0],
    [0, -1]
])
print(Q)
#inverse equivalence
V = inv(Q)
print(Q.T)
print(V)
#identity equivalence
I = Q.dot(Q.T)
print(I)
