#!/usr/bin/python3
"""Inverse
finds another matrix which when multiplied with the matrix results in an identity matrix"""
import numpy as np
from numpy.linalg import inv


A = np.array([
    [1.0, 2.0],
    [3.0, 4.0]
])
print(A)
print("-------------------")
B = inv(A)
print(B)
print("---------------------")
I = A.dot(B)
print(I)
