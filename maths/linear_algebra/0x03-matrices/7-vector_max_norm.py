#!/usr/bin/python3
"""vector max norm
Calculated by returning the max value of the vector"""
import numpy as np
from numpy.linalg import norm


#define vector
a = np.array([1, 2, 3])
print(a)
#calculate max norm
maxnorm = norm(a, np.inf)
print(maxnorm)
b = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(norm(b, np.inf))
#testing previous norms
print(norm(b))
print(norm(b, 1))
