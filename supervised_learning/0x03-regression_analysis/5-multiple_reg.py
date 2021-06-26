#!/usr/bin/python3
"""
Multiple linear regression
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('housing_data.csv')
print(df.head())
plt.figure(figsize=(10, 7))
plt.suptitle('Parameters vs Median value')
ax1 = plt.subplot(121)
ax1.scatter(df.LSTAT, df.MEDV, marker='*', c='k')
ax1.set_xlabel('% lower status of the population')
ax1.set_ylabel('Median Value in $1000s')
ax2 = plt.subplot(122, sharey=ax1)
ax2.scatter(df.RM, df.MEDV, marker='*', c='k')
ax2.get_yaxis().set_visible(False)
ax2.set_xlabel('average number of rooms per dwelling')
plt.savefig('multiple.png')

#modelling
model = LinearRegression()
model.fit(df.LSTAT.values.reshape((-1, 1)), df.MEDV.values.reshape((-1, 1)))
r2 = model.score(df.LSTAT.values.reshape((-1, 1)), df.MEDV.values.reshape((-1, 1)))
print(r2)

#trained using avg no. of rooms
model.fit(df.RM.values.reshape((-1, 1)), df.MEDV.values.reshape((-1, 1)))
r2 = model.score(df.RM.values.reshape((-1, 1)), df.MEDV.values.reshape((-1, 1)))
print(r2)

#prediction based on LSTAT and RM values
model.fit(df[['LSTAT', 'RM']], df.MEDV.values.reshape((-1, 1)))
r2 = model.score(df[['LSTAT', 'RM']], df.MEDV.values.reshape((-1, 1)))

print(r2)
