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
# virginica species lying within versicolor
df_test = df.iloc[134]
df = df.drop([134])
print(df_test)
plt.figure(figsize=(10, 7))
for name, group in df.groupby('Species'):
    plt.scatter(group['Sepal Length'], group['Petal Width'],
                label=name,
                marker=markers[name]['marker'],
                facecolors=markers[name]['facecolor'],
                edgecolor=markers[name]['edgecolor'])
plt.scatter(df_test['Sepal Length'], df_test['Petal Width'], label='Test Sample', c='k', marker='D')
plt.title('Species Classification Sepal Length vs Petal Width')
plt.xlabel('Sepal Length (mm)')
plt.ylabel('Petal Width (mm)')
plt.legend()
plt.savefig('5-iris_scatter.png')

# KNN classifier
model = KNN(n_neighbors=3)
print(model.fit(X=df[['Petal Width', 'Sepal Length']], y=df.Species))
print(model.score(X=df[['Petal Width', 'Sepal Length']], y=df.Species))
# predict the species of the test sample
print(model.predict(df_test[['Petal Width', 'Sepal Length']].values.reshape((-1,2)))[0])
print(df.iloc[134].Species)
