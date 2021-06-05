#!/usr/bin/python3
"""Determinant
is a scalar representation of the volume of the matrix"""
import numpy as np
from numpy.linalg import det


A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(A)
B = det(A)
print(B)
