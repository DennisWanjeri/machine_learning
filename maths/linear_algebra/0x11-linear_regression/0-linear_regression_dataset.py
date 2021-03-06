#!/usr/bin/python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot

data = np.array([
    [0.05, 0.12],
    [0.18, 0.22],
    [0.31, 0.35],
    [0.42, 0.38],
    [0.5, 0.49]
])
print(data)
x, y = data[:, 0], data[:, 1]
print(x)
print(y)
x = x.reshape((len(x), 1))
print("----------------")
print(x)
pyplot.scatter(x, y)
pyplot.savefig('1-reg.png')
