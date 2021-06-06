#!/usr/bin/python3
"""svd data reduction in scikit-learn"""
import numpy as np
from sklearn.decomposition import TruncatedSVD


A = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
])
print(A)
#create transform
svd = TruncatedSVD(n_components=2)
#fit transform
svd.fit(A)
#apply transform
result = svd.transform(A)
print(result)
