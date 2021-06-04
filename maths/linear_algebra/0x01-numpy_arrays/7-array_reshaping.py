#!/usr/bin/python3
"""reshaping arrays"""
import numpy as np


data = np.array([11, 22, 33, 44, 55])
print(data.shape)
print("------------------")

data_1 = np.array([
    [11, 22, 33],
    [44, 55, 66],
    [77, 88, 99]
])
print(data_1.shape)
print("-------------------------")
print(data.reshape((data.shape[0], 1)))
print("-----------------------")
print(data_1.reshape((data_1.shape[0], data_1.shape[1], 1)))
