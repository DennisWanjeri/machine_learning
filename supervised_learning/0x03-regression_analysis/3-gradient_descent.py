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

#r squared score
from sklearn.metrics import r2_score
r2_score(y_true, y_pred)
#linspace to get 20yr increments
x = np.linspace(df_group_year['Year'].min(), df_group_year['Year'].max(), 20)
print(x)
#trend_x
trend_x = np.ones((2, len(x)))
trend_x[0, :] = x
trend_x[1, :] = 1
trend_x /= trend_x.max()
print(trend_x)
trend_y = h_x(Theta, trend_x)

#plotting the trendline
plt.figure(figsize=(10, 7))
#Temp measurements
plt.scatter(df_group_year.index, df_group_year.AverageTemperature, label='Raw Data', c='k')
plt.plot(df_group_year.index, rolling, c='k', linestyle='--',
         label='{} year moving average'.format(window))
plt.plot(x, trend_y, c='k', label='Model: Predicted trendline')
plt.title('Mean Air Temperature Measurements')
plt.xlabel('Year')
plt.ylabel('Temperature (degC)')
plt.xticks(range(df_group_year.index.min(), df_group_year.index.max(), 10))
plt.legend()
plt.savefig('gradient_descent.png')
