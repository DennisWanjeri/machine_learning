#!/usr/bin/python3
"""Visualizing K-NN Boundaries"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier as KNN

df = pd.read_csv('iris-data.csv')
print(df.head())
labelled_species = [
    'Iris-setosa',
    'Iris-versicolor',
    'Iris-virginica'
]
# cool use of enumerate
for idx, label in enumerate(labelled_species):
    df.Species = df.Species.replace(label, idx)

print(df.head())
model = KNN(n_neighbors=3)
print(model.fit(X=df[['Sepal Length', 'Petal Width']], y=df.Species))
spacing = 0.1
petal_range = np.arange(df['Petal Width'].min() - 1, df['Petal Width'].max() + 1, spacing)
sepal_range = np.arange(df['Sepal Length'].min() - 1, df['Sepal Length'].max() + 1, spacing)
xx, yy = np.meshgrid(sepal_range, petal_range) # Create the mesh
print(xx)
print(yy)
pred_x = np.c_[xx.ravel(), yy.ravel()] # Concatenate the results
print(pred_x)
# create color maps
cmap_light = ListedColormap(['#F6A56F', '#6FF6A5', '#A56FF6'])
cmap_bold = ListedColormap(['#E6640E', '#0EE664', '#640EE6'])
markers = {
    'Iris-setosa': {'marker': 'x', 'facecolor': 'k', 'edgecolor': 'k'},
    'Iris-versicolor': {'marker': '*', 'facecolor': 'none', 'edgecolor':'k'},
    'Iris-virginica': {'marker': 'o', 'facecolor': 'none', 'edgecolor':'k'},
}
plt.figure(figsize=(10, 7))
for name, group in df.groupby('Species'):
    species = labelled_species[name]
    plt.scatter(group['Sepal Length'], group['Petal Width'],
                c=cmap_bold.colors[name],
                label=labelled_species[name],
                marker=markers[species]['marker']
            )
plt.title('Species Classification Sepal Length vs Petal Width')
plt.xlabel('Sepal Length (mm)')
plt.ylabel('Petal Width (mm)')
plt.legend()
plt.savefig('visualized_boundari.png')
