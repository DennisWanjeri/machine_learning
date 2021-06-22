#!/usr/bin/env python3
"""Linear regression using optimized gradient deacent"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('synth_temp.csv')
df = df.loc[df.Year > 1901]
df_group_year = df.groupby('Year').agg(np.mean)
window = 10
rolling = df_group_year.AverageTemperature.rolling(window).mean()
print(rolling.head(n=20))
df_group_year['Year'] = df_group_year.index

model = SGDRegressor(
    max_iter=100000,
    learning_rate='constant',
    eta0=1e-6,
    random_state=255,
    tol=1e-6,
    penalty='none'
)
x = df_group_year.Year / df_group_year.Year.max()
y_true = df_group_year.AverageTemperature.values.ravel()
model.fit(x.values.reshape((-1, 1)), y_true)
#predict values using trained model
y_pred = model.predict(x.values.reshape((-1, 1)))
r2_score(y_true, y_pred)

x = np.linspace(df_group_year['Year'].min(), df_group_year['Year'].max(), 20)
_x = x/x.max()
trend_y = model.predict(_x.reshape((-1, 1)))
#plotting
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
plt.savefig('sgdregressor.png')
