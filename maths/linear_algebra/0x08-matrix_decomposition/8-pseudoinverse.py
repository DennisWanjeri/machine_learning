#!/usr/bin/python3
"""Pseudoinverse
is the generalisation of the matrix inverse for rectangular matrices
since matrix inversion is not defined in rectangular matrices"""
from numpy.linalg import pinv
import numpy as np


A = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
    [0.7, 0.8]
])
print(A)
#calculate pseudoinverse
B = pinv(A)
print(B)
