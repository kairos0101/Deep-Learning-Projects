import numpy as np
import pandas as pd
import MNIST_def as MNIST_def

data = pd.read_csv("/Users/jaydemirandilla/VS_DeepLearning/SimpleMNIST/digit-recognizer/train.csv")

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

W1, b1, W2, b2 = MNIST_def.gradient_descent(X_train, Y_train, 0.10, 500, m)

MNIST_def.test_prediction(0, W1, b1, W2, b2, X_train, Y_train)
MNIST_def.test_prediction(1, W1, b1, W2, b2, X_train, Y_train)
MNIST_def.test_prediction(2, W1, b1, W2, b2, X_train, Y_train)
MNIST_def.test_prediction(3, W1, b1, W2, b2, X_train, Y_train)

dev_predictions = MNIST_def.make_predictions(X_dev, W1, b1, W2, b2)
print(MNIST_def.get_accuracy(dev_predictions, Y_dev))