import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from common.functions import *

x = np.ones(10)
y = np.zeros(10)
print(mse(x, y))
x = np.random.randn(10).reshape(2,5)
print(x)
print(softmax(x))

# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
#
# print(mnist.train.images.__class__)
# print(mnist.train.labels.shape)
# print(mnist.test.images.shape)
# print(mnist.test.labels.shape)