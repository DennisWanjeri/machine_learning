#!/usr/bin/python3
"""Triangular Matrix
a type of square that has all values in the upper_right or
lower-left of the matrix with the remaining elements filled
with zeroes"""
import numpy as np


M = np.array([
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]
])
print(M)
#lower triangular matrix
lower = np.tril(M)
print(lower)
#upper triangle matrix
upper = np.triu(M)
print(upper)
