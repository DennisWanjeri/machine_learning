#!/usr/bin/python3
"""converting a list to an ndarray"""
import numpy as np


#one dimensional list
data = [11, 22, 33, 44, 54]
data = np.array(data)
print(data)
print(type(data))

#two dimensional list
data_1 = [[11, 22, 33],
          [44, 55, 66],
          [77, 88, 99]
      ]
data_1 = np.array(data_1)
print(data_1)
print(type(data_1))
#indexing
print(data[0])
print(data_1[0,])
print(data_1[2, 2])
