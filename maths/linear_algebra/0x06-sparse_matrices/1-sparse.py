#!/usr/bin/python3
"""creating a sparse matrix"""
import numpy as np
from scipy.sparse import csr_matrix


A = np.array([
    [1, 0, 0, 1, 0, 0],
    [0, 0, 2, 0, 0, 1],
    [0, 0, 0, 2, 0, 0]
])
print(A)
#convert to sparse matrix(CSR method)
S = csr_matrix(A)
print(S)
B = S.todense()
print(B)
