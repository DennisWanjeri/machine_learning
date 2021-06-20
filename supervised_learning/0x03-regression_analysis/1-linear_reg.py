#!/usr/bin/env python3
"""Linear regression"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('synth_temp.csv')
df = df.loc[df.Year > 1901]
df_group_year = df.groupby('Year').agg(np.mean)
window = 10
rolling = df_group_year.AverageTemperature.rolling(window).mean()
print(rolling.head(n=20))
#plotting measurements by year along with mobing average signal
plt.figure(figsize=(10, 7))
#Temp measurements
plt.scatter(df_group_year.index, df_group_year.AverageTemperature, label='Raw Data', c='k')
plt.plot(df_group_year.index, rolling, c='k', linestyle='--',
        label='{} year moving average'.format(window))
plt.title('Mean Air Temperature Measurements')
plt.xlabel('Year')
plt.ylabel('Temperature (degC)')
plt.xticks(range(df_group_year.index.min(), df_group_year.index.max(), 10))
plt.legend()
plt.savefig('mean_annual_temp.png')
