#!/usr/bin/python3
"""calculating variance along rows and columns"""
import numpy as np


M = np.array([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6]
])
print(M)
#column variances
col = np.var(M, ddof=1, axis=0)
print(col)
#row variances
row = np.var(M, ddof=1, axis=1)
print(row)
