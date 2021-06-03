#!/usr/bin/python3
"""combining arrays using vstack and horizontal stack"""
import numpy as np


a = np.array([1, 2, 3])
print(a)
print("------------------")
b = np.array([4, 5, 6])
print(b)
print("----------+_---------")
#vertical stack
c = np.vstack((a, b))
print(c)
print(c.shape)
#horizontal stack
print("------++++----+++++++----")
d = np.hstack((a, b))
print(d)
print(d.shape)
