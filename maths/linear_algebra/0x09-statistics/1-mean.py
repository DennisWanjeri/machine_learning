#!/usr/bin/python3
"""calculating mean in matrices by changing axes values"""
import numpy as np


M = np.array([
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6]
])
print(M)
#column mean
col = np.mean(M, axis=0)
print(col)
#row mean
row = np.mean(M, axis=1)
print(row)
