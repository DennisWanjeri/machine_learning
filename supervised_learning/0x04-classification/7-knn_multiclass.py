#!/usr/bin/python3
"""using the K-NN model to classify MNIST dataset"""
import matplotlib
matplotlib.use('Agg')
import struct
import numpy as np
import gzip
import urllib.request
import matplotlib.pyplot as plt
from array import array
from sklearn.neighbors import KNeighborsClassifier as KNN

# loading MNIST data
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

# visualizing a sample of the data
plt.figure(figsize=(10, 7))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img[i], cmap='gray')
    plt.title('{}'.format(labels[i]))
    plt.axis('off')

plt.savefig('7-sample.png')

# selaction
selection = np.random.choice(len(img), 5000)
selected_images = img[selection]
selected_labels = labels[selection]
print(img.shape)
print(selected_images.shape)
selected_images = selected_images.reshape((-1, rows * cols))
print(selected_images.shape)

model = KNN(n_neighbors=3)
print(model.fit(X=selected_images, y=selected_labels))
print(model.score(X=selected_images, y=selected_labels))

# display first two predictions
print(model.predict(selected_images)[:2])
plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
plt.imshow(selected_images[0].reshape((28, 28)), cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(selected_images[1].reshape((28, 28)), cmap='gray')
plt.axis('off')
plt.savefig("7-predictions.png")

# scoring against test data
model.score(X=img_test.reshape((-1, rows * cols)), y=labels_test)
