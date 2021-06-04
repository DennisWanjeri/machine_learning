#!/usr/bin/python3
"""array slicing"""
import numpy as np


data = np.array([11, 22, 33, 44, 55])
print(data[:])
print(data[0:1])

data = np.array([
    [11, 22, 33],
    [44, 55, 66],
    [77, 88, 99]
])

x, y = data[:, :-1], data[:, -1]
print(x)
print(y)
