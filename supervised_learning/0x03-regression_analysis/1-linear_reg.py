#!/usr/bin/env python3
"""Linear regression"""
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

#Fitting a linear model using the Least Squares Method
model = LinearRegression()
print(model)
model.fit(df_group_year.index.values.reshape((-1, 1)), df_group_year.AverageTemperature)
print("m = {}".format(model.coef_[0]))
print("c = {}".format(model.intercept_))
print('\nModel Definition')
print('y = {}x + {}'.format(model.coef_[0], model.intercept_))
trend_x = np.array([
    df_group_year.index.values.min(),
    df_group_year.index.values.mean(),
    df_group_year.index.values.max()
])
trend_y = model.predict(trend_x.reshape((-1, 1)))
print(trend_y)
#plotting measurements by year along with moving average signal
plt.figure(figsize=(10, 7))
#Temp measurements
plt.scatter(df_group_year.index, df_group_year.AverageTemperature, label='Raw Data', c='k')
plt.plot(df_group_year.index, rolling, c='k', linestyle='--',
         label='{} year moving average'.format(window))
plt.plot(trend_x, trend_y, c='k', label='Model: Predicted trendline')
plt.title('Mean Air Temperature Measurements')
plt.xlabel('Year')
plt.ylabel('Temperature (degC)')
plt.xticks(range(df_group_year.index.min(), df_group_year.index.max(), 10))
plt.legend()
plt.savefig('mean_annual_temp.png')
