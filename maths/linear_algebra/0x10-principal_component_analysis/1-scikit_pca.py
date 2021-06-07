#!/usr/bin/python3
"""Principal component analysis in scikit-learn"""
import numpy as np
from sklearn.decomposition import PCA


A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
print(A)
#create the transform
pca = PCA(2)
#fit transform
pca.fit(A)
#access values and vectors
print(pca.components_)
print(pca.explained_variance_)
#transform data
B = pca.transform(A)
print(B)
