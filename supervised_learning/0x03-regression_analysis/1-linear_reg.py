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
#r2 value
r2 = model.score(df_group_year.index.values.reshape((-1, 1)), df_group_year.AverageTemperature)
print("r2 score = {}".format(r2))

#Dummy variables
df_group_year['Year'] = df_group_year.index
df_group_year['Gt_1960'] = [0 if year < 1960 else 10 for year in df_group_year.Year]
print(df_group_year.head(n=2))
df_group_year['Gt_1945'] = [0 if year < 1945 else 10 for year in df_group_year.Year]
print(df_group_year.head(n=2))
print(df_group_year.tail(n=2))
model.fit(df_group_year[['Year', 'Gt_1960', 'Gt_1945']], df_group_year.AverageTemperature)
r2 = model.score(df_group_year[['Year', 'Gt_1960', 'Gt_1945']], df_group_year.AverageTemperature)
print("r2 = {}".format(r2))
#use linspace to get a range of 20 yr increments
x = np.linspace(df_group_year['Year'].min(), df_group_year['Year'].max(), 20)
print(x)
trend_x = np.zeros((20, 3))
trend_x[:, 0] = x
trend_x[:, 1] = [10 if _x > 1960 else 0 for _x in x]
trend_x[:, 2] = [10 if _x > 1945 else 0 for _x in x]
print(trend_x)
trend_y = model.predict(trend_x)
print(trend_y)
#plotting measurements by year along with moving average signal
plt.figure(figsize=(10, 7))
#Temp measurements
plt.scatter(df_group_year.index, df_group_year.AverageTemperature, label='Raw Data', c='k')
plt.plot(df_group_year.index, rolling, c='k', linestyle='--',
         label='{} year moving average'.format(window))
plt.plot(trend_x[:, 0], trend_y, c='k', label='Model: Predicted trendline')
plt.title('Mean Air Temperature Measurements')
plt.xlabel('Year')
plt.ylabel('Temperature (degC)')
plt.xticks(range(df_group_year.index.min(), df_group_year.index.max(), 10))
plt.legend()
plt.savefig('mean_annual_temp.png')
