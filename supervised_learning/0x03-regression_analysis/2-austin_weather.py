#!/usr/bin/python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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
model = LinearRegression()
df_first_year.loc[:, 'DayOfYear'] = [i + 1 for i in df_first_year.index]
model.fit(df_first_year.DayOfYear.values.reshape((-1, 1)), df_first_year.TempAvgF)
print("m = {}".format(model.coef_[0]))
print("c = {}".format(model.intercept_))

print('\nModel Definition')
print('y = {}x + {}'.format(model.coef_[0], model.intercept_))
trend_x = np.array([
    df_first_year.index.values.min(),
    df_first_year.index.values.mean(),
    df_first_year.index.values.max()
])
trend_y = model.predict(trend_x.reshape((-1, 1)))

print(trend_y)
#plotting
plt.figure(figsize=(10, 7))
plt.scatter(range(1, 366), df_first_year.TempAvgF, label='Raw Data')
plt.plot(range(1, 366), rolling, c='r', label='{} day moving average'.format(window))
plt.plot(trend_x, trend_y, c='k', label='Model: Predicted trendline')
plt.title('Daily Mean Temperature Measurements')
plt.xlabel('Day')
plt.ylabel('Temperature (degF)')
plt.xticks(range(1, 366, 10))
plt.legend()
plt.savefig('newscatter.jpg')
