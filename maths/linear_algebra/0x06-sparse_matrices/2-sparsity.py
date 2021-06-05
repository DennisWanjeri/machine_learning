#!/usr/bin/python3
"""calculating the sparsity of a matrix"""
import numpy as np
from numpy import count_nonzero


A = np.array([
    [1, 0, 0, 1, 0, 0],
    [0, 0, 2, 0, 0, 1],
    [0, 0, 0, 2, 0, 0]
])
print(A)
sparsity = 1.0 - count_nonzero(A) / A.size
print(sparsity)
