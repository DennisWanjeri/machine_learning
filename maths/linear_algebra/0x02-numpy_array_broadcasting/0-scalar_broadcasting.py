#!/usr/bin/python3
"""multiplying a scalar with a one dimensional array"""
import numpy as np


a = [1, 2, 3]
#define scalar b
b = 2
a = np.array(a)
c = a + b
#does [a1 + b1, a2 + b2, a3 + b3] where b1, b2, b3 are duplicates
print(c)

"""two dimensional array scalar broadcasting"""
print("------------------------")
A = np.array([
    [1, 2, 3],
    [1, 2, 3]
])
b = 2
#broadcast
C = A + b
print(C)

"""one dimensional array can multiply a two dimensional array"""
print("-------------------")
b = np.array([1, 2, 3])
#to multiply, an array B is created which has two rows, dup of 1st row
C = A + b
print(C)
