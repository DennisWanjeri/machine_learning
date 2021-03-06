#!/usr/bin/python3
"""Iris classification using a CART Decision Tree"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import graphviz
from sklearn.tree import export_graphviz

df = pd.read_csv('iris-data.csv')
print(df.head())

# take a random sample of 10 rows
np.random.seed(10)
samples = np.random.randint(0, len(df), 10)
df_test = df.iloc[samples]
df = df.drop(samples)

model = DecisionTreeClassifier()
model = model.fit(df[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']], df.Species)
print(model.score(df[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']], df.Species))
print(model.score(df_test[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']], df_test.Species))

dot_data = export_graphviz(model, out_file='dot_data.dot')
graph = graphviz.Source(dot_data)
