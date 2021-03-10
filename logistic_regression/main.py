from logistic_regression.classifier import IRLS, SGD, GD
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# TODO use simple data with known beta, eg lab3/task3

iris = load_iris()
X = np.array(iris.data)
y = np.array(iris.target)
y[y == 2] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

irls = IRLS(max_iter=100)
irls.train(X_train, y_train)
1 / (1 + np.exp(-X @ irls.beta))
irls.beta

gd = GD(learning_rate=0.01)
gd.train(X_train, y_train)
1 / (1 + np.exp(-X @ gd.beta))
gd.beta

sgd = SGD(learning_rate=0.01)
sgd.train(X_train, y_train)
1 / (1 + np.exp(-X @ sgd.beta))
sgd.beta
