from logistic_regression.classifier import IRLS, SGD, GD
from logistic_regression.metric import Metric
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# TODO use simple data with known beta, eg lab3/task3

iris = load_iris()
X = np.array(iris.data)
y = np.array(iris.target)
y[y == 2] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

irls = IRLS(max_iter=100).train(X_train, y_train)
print(f'IRLS accuracy: {irls.score(X_test, y_test, Metric.Acc)} and beta: {irls.beta}')

gd = GD(learning_rate=0.01).train(X_train, y_train)
print(f'GD accuracy: {gd.score(X_test, y_test, Metric.Acc)} and beta: {gd.beta}')

sgd = SGD(learning_rate=0.01).train(X_train, y_train)
print(f'SGD accuracy: {sgd.score(X_test, y_test, Metric.Acc)} and beta: {sgd.beta}')
