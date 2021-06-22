#!/usr/bin/env python3
"""Linear regression using gradient deacent"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('synth_temp.csv')
df = df.loc[df.Year > 1901]
df_group_year = df.groupby('Year').agg(np.mean)
window = 10
rolling = df_group_year.AverageTemperature.rolling(window).mean()
print(rolling.head(n=20))

def h_x(weights, x):
    return np.dot(weights, x).flatten()
df_group_year['Year'] = df_group_year.index
x = np.ones((2, len(df_group_year)))
x[0, :] = df_group_year.Year
x[1, :] = 1
#normalizing the values to be between 1 and 0
x /= x.max()
print(x[:, :5])
np.random.seed(255)# ensure same starting random values
Theta = np.random.randn(2).reshape((1, 2)) * 0.1
print(Theta)
y_true = df_group_year.AverageTemperature.values

#define the cost function
def j_theta(pred, true):
    """cost function"""
    return np.mean((pred - true) ** 2)# mean squared error
#define learning rate
gamma = 1e-6
#implements a step of gradient descent

def update(pred, true, x, gamma):
    """takes the predicted and true values and
    returns value to be added to the weights"""
    return gamma * np.sum((true - pred) * x, axis = 1)

#max no. of iterations
max_epochs = 100000

#an initial prediction
y_pred = h_x(Theta, x)
print("Initial cost J(Theta) = {}".format(j_theta(y_pred, y_true)))

#manual update
Theta += update(y_pred, y_true, x, gamma)
y_pred = h_x(Theta, x)
print("Initial cost J(Theta) = {}".format(j_theta(y_pred, y_true)))

error_hist = []
epoch_hist = []
for epoch in range(max_epochs):
    Theta += update(y_pred, y_true, x, gamma)
    y_pred = h_x(Theta, x)

    if (epoch % 10) == 0:
        _err = j_theta(y_pred, y_true)
        error_hist.append(_err)
        epoch_hist.append(epoch)
        print("epoch:{} J(Theta) = {}".format(epoch, _err))

#visualizing training history
plt.figure(figsize=(10, 7))
plt.plot(epoch_hist, error_hist)
plt.title('Trqining History')
plt.xlabel('epoch')
plt.ylabel('Error')
plt.savefig('model_training.png')
