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


# selecting a random subset of overall data
np.random.seed(0)
selection = np.random.choice(len(img), 5000)
selected_images = img[selection]
selected_labels = labels[selection]

selected_images = selected_images.reshape((-1, rows * cols))
print(selected_images.shape)

selected_images = selected_images / 255.0
img_test = img_test / 255.0

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=500,tol=0.1)
print(model.fit(X=selected_images, y=selected_labels))
print(model.score(X=selected_images, y=selected_labels))
print(model.predict(selected_images)[:2])

plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
plt.imshow(selected_images[0].reshape((28, 28)), cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(selected_images[1].reshape((28, 28)), cmap='gray')
plt.axis('off')
plt.savefig('1_4_prediction.png')

#probability score
print(model.predict_proba(selected_images)[0])

print(model.score(X=img_test.reshape((-1, rows * cols)), y=labels_test))
