#!/usr/bin/python3
"""broadcasting limitation"""
import numpy as np


A = np.array([
    [1, 2, 3],
    [1, 2, 3]
])
print (A.shape)
b = np.array([1, 2])
print(b.shape)
#attempt broadcast, fail coz columns are not equal
C = A + b
print(C)
