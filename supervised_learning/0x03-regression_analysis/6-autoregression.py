#!/usr/bin/python3
"""
Autoregression
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR

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

print(df.close[:10].values)
print(df.close[:10].shift(3).values)
model = AR(df.close)
model_fit = model.fit()
print("Lag: {}".format(model_fit.k_ar))
print("coefficients: {}".format(model_fit.params))
predictions = model_fit.predict(start=36, end=len(df) + 500)
print(predictions[:10].values)
#plotting
plt.figure(figsize=(10, 7))
plt.plot(df.close.values, label='original dataset')
plt.plot(predictions, c='g', linestyle=':', label='predictions')
yrs = [yr for yr in df.Year.unique() if (int(yr[-2:]) % 5 == 0)]
plt.xticks(np.arange(0, len(df), len(df) // len(yrs)), yrs)
plt.title('S&P 500 Daily Closing Price')
plt.xlabel('Year')
plt.ylabel('Price ($)')
plt.savefig('1-spx.png')
plt.legend()
#auto_correlation plot
plt.figure(figsize=(10,7))
pd.plotting.autocorrelation_plot(df.close)
plt.savefig('autocorrelation.png')
#visualizing the lag
plt.figure(figsize=(10, 7))
ax = pd.plotting.autocorrelation_plot(df.close)
ax.set_ylim([-0.1, 0.1])
plt.savefig('visualized.png')
#autocorrelation plot
plt.figure(figsize=(10,7))
pd.plotting.lag_plot(df.close, lag=4000)
plt.savefig('lag_plot.png')
