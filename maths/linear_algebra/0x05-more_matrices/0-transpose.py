#!/usr/bin/python3
"""Transpose
creates a new matrix with the number of columns and rows flipped"""
import numpy as np


A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
print(A)
#calculate transpose
C = A.T
print(C)
