#!/usr/bin/python3
"""Logistic Regression as a Classifier"""
import matplotlib
matplotlib.use('Agg')
import struct
import numpy as np
import gzip
import urllib.request
import matplotlib.pyplot as plt
from array import array
from sklearn.linear_model import LogisticRegression

"""request = urllib.request.urlopen('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')

with open('train-images-idx3-ubyte.gz', 'wb') as f:
    f.write(request.read())

request = urllib.request.urlopen('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
with open('t10k-images-idx3-ubyte.gz', 'wb') as f:
    f.write(request.read())

request = urllib.request.urlopen('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
with open('train-labels-idx1-ubyte.gz', 'wb') as f:
    f.write(request.read())

request = urllib.request.urlopen('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
with open('t10k-labels-idx1-ubyte.gz', 'wb') as f:
    f.write(request.read())
"""
with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
    magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
    img = np.array(array("B", f.read())).reshape((size, rows, cols))

with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    labels = np.array(array("B", f.read()))

with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f:
    magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
    img_test = np.array(array("B", f.read())).reshape((size, rows, cols))

with gzip.open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    labels_test = np.array(array("B", f.read()))

plt.figure(figsize=(10, 7))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img[i], cmap='gray')
    plt.title("{}".format(labels[i]))
    plt.axis('off')
plt.savefig('images.png')

samples_0_1 = np.where((labels == 0) | (labels == 1))[0]
images_0_1 = img[samples_0_1]
labels_0_1 = labels[samples_0_1]

samples_0_1_test = np.where((labels_test == 0) | (labels_test == 1))
images_0_1_test = img_test[samples_0_1_test].reshape((-1, rows * cols))
labels_0_1_test = labels_test[samples_0_1_test]

sample_0 = np.where((labels == 0))[0][0]
sample_1 = np.where((labels == 1))[0][0]
plt.figure(figsize=(10, 7))
plt.imshow(img[sample_0], cmap='gray')
plt.axis('off')
plt.savefig('0_images.png')

plt.figure(figsize=(10, 7))
plt.imshow(img[sample_1], cmap='gray')
plt.axis('off')
plt.savefig('1-image.png')

# rearranging images to vector format
images_0_1 = images_0_1.reshape((-1, rows * cols))
print(images_0_1.shape)
model = LogisticRegression(solver='liblinear')
print(model.fit(X=images_0_1, y=labels_0_1))
# performance
r = model.score(X=images_0_1, y=labels_0_1)
print(r)
print(model.predict(images_0_1) [:2])
# probabilities produced by the model for the training set
print(model.predict_proba(images_0_1)[:2])

#computing performance against test check
print(model.score(X=images_0_1_test, y=labels_0_1_test))
