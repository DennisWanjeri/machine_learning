#!/usr/bin/python3
"""variance and standard deviation"""
#calculating vector variance
import numpy as np


v = np.array([1, 2, 3, 4, 5, 6])
print(v)
#calculate variance
result = np.var(v, ddof=1)
print(result)
