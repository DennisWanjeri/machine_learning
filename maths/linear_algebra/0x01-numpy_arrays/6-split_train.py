#!/usr/bin/python3
"""splitting loaded dataset into separate train and test sets"""
import numpy as np


data = np.array([
    [11, 22, 33],
    [44, 55, 66],
    [77, 88, 99]
])

split = 2
train, split = data[:split, :], data[split:, :]
print(train)
print("-----------------------")
print(split)
