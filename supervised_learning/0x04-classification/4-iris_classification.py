#!/usr/bin/python3
"""iris classification using logistic regression"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('iris-data.csv')
print(df.head())
# feature engineering
markers = {
    'Iris-setosa': {'marker': 'x'},
    'Iris-versicolor': {'marker': '*'},
    'Iris-virginica': {'marker': 'o'}
}
plt.figure(figsize=(10, 7))
for name, group in df.groupby('Species'):
    plt.scatter(group['Sepal Width'], group['Petal Length'],
                label=name,
                marker=markers[name]['marker'],
            )
plt.title('Species Classification Sepal Width vs Petal Length')
plt.xlabel('Sepal Width (mm)')
plt.ylabel('Petal Length (mm)')
plt.legend()
plt.savefig('iris_scatter.png')
