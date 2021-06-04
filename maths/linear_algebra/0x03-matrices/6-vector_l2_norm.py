#!/usr/bin/python3
"""calculating norm using Vector L2 Norm
calculated as the square root of the sum of square of individual squares"""
import numpy as np
from numpy.linalg import norm


a = np.array([1, 2, 3])
print(a)
l2 = norm(a)
print(l2)
