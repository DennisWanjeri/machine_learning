#!/usr/bin/python3
from numpy.linalg import qr, inv
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
#split into inputs and outputs
x, y = data[:, 0], data[:, 1]
x = x.reshape((len(x), 1))
#factorize
Q, R = qr(x)
b = inv(R).dot(Q.T).dot(y)
print(b)
#predict using coefficients
yhat = x.dot(b)
#plot data and predictions
pyplot.scatter(x, y)
pyplot.plot(x, yhat, color='red')
pyplot.savefig('2-qr_reg.png')
