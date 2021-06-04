#!/usr/bin/python3
"""Diagonal Matrix
one where values outside of main diagonal have a zero value"""
import numpy as np


M = np.array([
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]
])
print(M)
print("-------------")
#extract diagonal.vector
d = np.diag(M)
print(d)
print("---------------")
#create diagonal matrix from vector
D = np.diag(d)
print(D)
