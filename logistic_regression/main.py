from logistic_regression.classifier import IRLS, SGD, GD
from logistic_regression.metric import Metric
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression.stopper import StopCondition


def gen_data(b0, b1, b2, n):
    x = []
    y = []
    for i in range(n):
        x1 = np.random.normal(0, 1)
        x2 = np.random.normal(0, 1)
        x.append([x1, x2])
        p = 1/(1 + np.exp(-(b0 + b1*x1 + b2*x2)))
        y.append(1 if np.random.uniform(0, 1) < p else 0)
    x = np.array(x)
    y = np.array(y)
    return x, y


def test_data(X, y, max_iter):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    irls = IRLS(max_iter=max_iter, eps=0.0001).train(X_train, y_train)
    print(f'IRLS accuracy: {irls.score(X_test, y_test, Metric.Acc)} and beta: {irls.beta}')

    gd = GD(max_iter=max_iter, learning_rate=0.01).train(X_train, y_train)
    print(f'GD accuracy: {gd.score(X_test, y_test, Metric.Acc)} and beta: {gd.beta}')

    sgd = SGD(max_iter=max_iter, learning_rate=0.01).train(X_train, y_train)
    print(f'SGD accuracy: {sgd.score(X_test, y_test, Metric.Acc)} and beta: {sgd.beta}')

    # ploting log-likelihood
    plt.plot(irls.log_likelihood, label="IRLS")
    plt.plot(gd.log_likelihood, label="GD")
    plt.plot(sgd.log_likelihood, label="SGD")
    plt.legend()
    plt.show()


def log_likelihood(y, pred):
    return np.log(pred) @ y.transpose() + np.log(1 - pred) @ (1 - y).transpose()


iris = load_iris()
X = np.array(iris.data)
y = np.array(iris.target)
y[y == 2] = 1

print("\n\nIris:\n")
test_data(X, y, max_iter=1000)


X, y = gen_data(0.5, 1, -2, 1000)

print("\n\nSyntetic(0.5, 1, -2):\n")
test_data(X, y, max_iter=1000)

# X, y = gen_data(0.5, 1, -2, 1000)
# gd = GD(max_iter=10000000000, learning_rate=0.01, stop_condition=StopCondition.LogLikelihood, max_iter_no_imp=10).train(X, y)
# plt.plot(gd.log_likelihood, label="GD")
# plt.legend()
# plt.show()