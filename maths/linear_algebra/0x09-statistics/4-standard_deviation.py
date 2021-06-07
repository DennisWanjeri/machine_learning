#!/usr/bin/python3
"""matrix standard deviation"""
import numpy as np


M = np.array([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6]
])
print(M)
#column standard deviation
col = np.std(M, ddof=1, axis=0)
print(col)
#row standard deviation
row = np.std(M, ddof=1, axis=1)
print(row)
