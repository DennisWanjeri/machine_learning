#!/usr/bin/python3
"""KNN Classification"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN

df = pd.read_csv('iris-data.csv')
print(df.head())

markers = {
    'Iris-setosa': {'marker': 'x', 'facecolor': 'k', 'edgecolor': 'k'},
    'Iris-versicolor': {'marker': '*', 'facecolor': 'none', 'edgecolor':
                        'k'},
    'Iris-virginica': {'marker': 'o', 'facecolor': 'none', 'edgecolor':
                       'k'},
    }
plt.figure(figsize=(10, 7))
for name, group in df.groupby('Species'):
    plt.scatter(group['Sepal Length'], group['Petal Width'],
                label=name,
                marker=markers[name]['marker'],
                facecolors=markers[name]['facecolor'],
                edgecolor=markers[name]['edgecolor'])
plt.title('Species Classification Sepal Length vs Petal Width')
plt.xlabel('Sepal Length (mm)')
plt.ylabel('Petal Width (mm)')
plt.legend()
plt.savefig('5-iris_scatter.png')
