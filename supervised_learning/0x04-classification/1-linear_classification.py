#!/usr/bin/python3
"""Linear Regression as a classifier"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('linear_classifier.csv')
print(df.head())

plt.figure(figsize=(10,7))
for label, label_class in df.groupby('labels'):
    plt.scatter(label_class.values[:, 0], label_class.values[:, 1],
                label="class {}".format(label), marker=label, c='k')
plt.legend()
plt.title('Linear Classifier')
plt.savefig('linear_classifier.png')
