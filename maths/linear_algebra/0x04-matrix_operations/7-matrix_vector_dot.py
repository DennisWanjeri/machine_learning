#!/usr/bin/python3
"""matrix vector dot product operation"""
import numpy as np


#can be multiplied together as long as m x n, n x p rule is satisfied
A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

B = np.array([0.5, 0.5])

C = A.dot(B)
print(C)
