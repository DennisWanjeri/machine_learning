#!/usr/bin/python3
"""vector correlation
covariance normalized to a score between -1 and 1"""
from numpy import corrcoef
import numpy as np


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(x)
y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1])
print(y)
#calculate correlation
corr = corrcoef(x, y)[0, 1]
print(corr)
