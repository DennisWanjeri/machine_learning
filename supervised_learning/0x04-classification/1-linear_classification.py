#!/usr/bin/python3
"""Linear Regression as a classifier"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.linear_model import LinearRegression

df = pd.read_csv('linear_classifier.csv')
print(df.head())

#fit a linear regression model
model = LinearRegression()
model.fit(df.x.values.reshape((-1, 1)), df.y.values.reshape((-1, 1)))
print("y = {}x + {}".format(model.coef_[0][0], model.intercept_[0]))
#plot the trendline
trend = model.predict(np.linspace(0, 10).reshape((-1, 1)))
#make predictions
y_pred = model.predict(df.x.values.reshape((-1, 1)))
pred_labels = []
for _y, _y_pred in zip(df.y, y_pred):
    if _y < _y_pred:
        pred_labels.append('o')
    else:
        pred_labels.append('x')
df['Pred_Labels'] = pred_labels
print(df.head())
plt.figure(figsize=(10,7))
plt.plot(np.linspace(0, 10), trend, c='k', label='Trendline')
for label, label_class in df.groupby('labels'):
    plt.scatter(label_class.values[:, 0], label_class.values[:, 1],
                label="class {}".format(label), marker=label, c='k')
plt.legend()
plt.title('Linear Classifier')
plt.savefig('linear_classifier.png')

#plotting the points with corresponding ground truth labels
plt.figure(figsize=(10, 7))
for idx, label_class in df.iterrows():
    if label_class['labels'] != label_class['Pred_Labels']:
        label = 'D'
        s = 70
    else:
        label = label_class['labels']
        s = 50
    plt.scatter(label_class.values[0], label_class.values[1],
                label = "class {}".format(label), marker=label, c='k', s= s)
plt.plot(np.linspace(0, 10), trend, c='k', label='Trendline')
plt.title("Linear Classifier")
incorrect_class = mlines.Line2D([], [], color='k', marker='D',
                                markersize=10, label='incorrect Classification')
#plt.legend(handles=[incorrect_class])
plt.savefig('incorrect_class.png')
