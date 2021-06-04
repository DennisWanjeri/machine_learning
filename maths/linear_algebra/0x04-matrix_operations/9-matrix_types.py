#!/usr/bin/python3
"""This module illustrates different kinds of matrices"""
import numpy as np


"""Square Matrix
is a matrix where number of rows is equivalent to no. of columns"""
square = np.ones([3, 3])
print(square)

"""Symmetric Matrix
where top-right triangle is the same as bottom-left triangle"""
sym = np.ones([5, 5])
print(sym)
