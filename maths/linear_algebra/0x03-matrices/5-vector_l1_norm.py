#!/usr/bin/python3
"""L1 norm magnitude calculation
L1 norm is calculates as the sum of the absolute vwctor values"""
import numpy as np
from numpy.linalg import norm


a = np.array([1, 2, 3])
print(a)

#calculate the norm
l1 = norm(a, 1)
print(l1)
