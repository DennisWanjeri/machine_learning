#!/usr/bin/python3
"""Rank
is the estimate of the no. of linearly independent rows or columns in a column"""
#a rank of 1 denotes matrix spans a line
#rank of 0 denotes matrix spans a point
#rank of 2 denotes a span of a two dimension
import numpy as np
from numpy.linalg import matrix_rank


v1 = np.array([0, 0, 0])
print(v1)
vr1 = matrix_rank(v1)
print(vr1)
v2 = np.array([1, 2, 3])
print(v2)
print(matrix_rank(v2))

M = np.array([
    [1, 2],
    [1, 2]
])
print(M)
mr = matrix_rank(M)
print(mr)

M2 = np.array([
    [1, 2],
    [3, 4]
])
print(M2)
m2r = matrix_rank(M2)
print(m2r)
