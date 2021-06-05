#!/usr/bin/python3
"""Trace
Sum of the values on the main diagonal of the matrix(top-left to bottom-right"""
import numpy as np


A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(A)
B = np.trace(A)
print(B)
