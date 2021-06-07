#!/usr/bin/python3
"""Covariance Matrix
is a square and symmetric matrix that describes the covariance between two or more random variables"""
import numpy as np
from numpy import cov


x = np.array([
    [1, 5, 8],
    [3, 5, 11],
    [2, 4, 9],
    [3, 6, 10],
    [1, 5, 10]
])
print(x)
#calculate covariance matrix
Sigma = cov(x.T)
print(Sigma)
