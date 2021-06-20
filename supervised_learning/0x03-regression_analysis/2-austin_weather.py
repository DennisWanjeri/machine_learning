#!/usr/bin/python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('austin_weather.csv')
df = df[['Date', 'TempAvgF']]
#extracting the date
df['Year'] = [int(dt[:4]) for dt in df.Date]
print(df.head())
#extract month
df['Month'] = [int(dt[5:7]) for dt in df.Date]
print(df.head())
#copy first yrs data worth in a dataframe
df_first_year = df[:365]
print(df_first_year.head())
#compute a 20-day moving average filter
window = 20
rolling = df_first_year.TempAvgF.rolling(window).mean()
print(rolling.head(n=30))
#plotting
fig = plt.figure(figsize=(10, 7))
axes = fig.add_axes([1, 1, 1, 1])
plt.scatter(range(1, 366), df_first_year.TempAvgF, label='Raw Data')
plt.plot(range(1, 366), rolling, c='r', label='{} day moving average'.format(window))
axes.set_title('Daily Mean Temperature Measurements')
axes.set_xlabel('Day')
axes.set_ylabel('Temperature (degF)')
axes.set_xticks(range(1, 366), 10)
axes.legend()
fig.savefig('scatter.png')
