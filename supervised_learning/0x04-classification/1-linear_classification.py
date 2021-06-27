#!/usr/bin/python3
"""Linear Regression as a classifier"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('linear_classifier.csv')
print(df.head())

#fit a linear regression model
model = LinearRegression()
model.fit(df.x.values.reshape((-1, 1)), df.y.values.reshape((-1, 1)))
print("y = {}x + {}".format(model.coef_[0][0], model.intercept_[0]))
#plot the trendline
trend = model.predict(np.linspace(0, 10).reshape((-1, 1)))
plt.figure(figsize=(10,7))
plt.plot(np.linspace(0, 10), trend, c='k', label='Trendline')
for label, label_class in df.groupby('labels'):
    plt.scatter(label_class.values[:, 0], label_class.values[:, 1],
                label="class {}".format(label), marker=label, c='k')
plt.legend()
plt.title('Linear Classifier')
plt.savefig('linear_classifier.png')
