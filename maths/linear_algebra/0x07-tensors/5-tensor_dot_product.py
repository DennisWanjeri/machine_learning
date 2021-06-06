#!/usr/bin/python3
"""tensor dot product"""
import numpy as np


A = np.array([1, 2])
B = np.array([3, 4])

C = np.tensordot(A, B, axes=0)
print(C)
