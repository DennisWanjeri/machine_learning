#!/usr/bin/python3
"""
Autoregression
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('spx.csv')
print(df.head())
yr = []
for x in df.date:
    x = int(x[-2:])
    if x < 10:
        x = "200{}".format(x)
    elif x < 20:
        x = "20{}".format(x)
    else:
        x = "19{}".format(x)
    yr.append(x)

df['Year'] = yr
print(df.head(n=20))

#plotting
plt.figure(figsize=(10, 7))
plt.plot(df.close.values)
yrs = [yr for yr in df.Year.unique() if (int(yr[-2:]) % 5 == 0)]
plt.xticks(np.arange(0, len(df), len(df) // len(yrs)), yrs)
plt.title('S&P 500 Daily Closing Price')
plt.xlabel('Year')
plt.ylabel('Price ($)')
plt.savefig('1-spx.png')
