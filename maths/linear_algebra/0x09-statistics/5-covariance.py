#!/usr/bin/python3
"""Vector Covariance
it describes how the two variables change together"""
import numpy as np


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(x)
y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1])
print(y)
Sigma = np.cov(x, y)[0, 1]
print(Sigma)
