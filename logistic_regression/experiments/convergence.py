from logistic_regression.classifier import IRLS, SGD, GD, MiniBatchGD
from logistic_regression.metric import Metric
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


def test_learning_rate(X, y, classifier, learning_rates, **kwargs):
    for learning_rate in learning_rates:
        model = classifier(learning_rate=learning_rate, **kwargs).train(X, y)
        plt.plot(model.log_likelihood, label=str(learning_rate))
    plt.title(f"Learning rates in {classifier.__name__}")
    plt.xlabel("Iteration number")
    plt.ylabel("Log likelihood")
    plt.legend()


def test_IRLS_regularization(X, y, epsilons, **kwargs):
    for eps in epsilons:
        model = IRLS(eps=eps, **kwargs).train(X, y)
        plt.plot(model.log_likelihood[0:], label=str(eps))
    plt.title(f"Regularization param in IRLS")
    plt.xlabel("Iteration number")
    plt.ylabel("Log likelihood")
    plt.legend()


plt.subplots(2, 2)
X, y = gen_data(0.5, 1, -2, 1000)
learning_rates = ([0.02, 0.05] + [1/(10**i) for i in range(1, 4)])
learning_rates.sort(reverse=True)
max_iter = 2000
plt.subplot(2, 2, 1)
test_learning_rate(X, y, GD, learning_rates, max_iter=max_iter)
plt.subplot(2, 2, 2)
test_learning_rate(X, y, SGD, learning_rates, max_iter=max_iter)
plt.subplot(2, 2, 3)
test_learning_rate(X, y, MiniBatchGD, learning_rates, max_iter=max_iter)

max_iter = 20
epsilons = [1/10**i for i in range(1, 6)]
plt.subplot(2, 2, 4)
test_IRLS_regularization(X, y, epsilons, max_iter=max_iter)
plt.show()


# check results with stop condition

plt.subplots(2, 2)
X, y = gen_data(0.5, 1, -2, 1000)
learning_rates = ([0.02, 0.05] + [1/(10**i) for i in range(1, 4)])
learning_rates.sort(reverse=True)
max_iter = 2000
plt.subplot(2, 2, 1)
test_learning_rate(X, y, GD, learning_rates, max_iter=max_iter, stop_condition=StopCondition.WorseThanWorst, max_iter_no_imp=100)
plt.subplot(2, 2, 2)
test_learning_rate(X, y, SGD, learning_rates, max_iter=max_iter, stop_condition=StopCondition.WorseThanWorst, max_iter_no_imp=100)
plt.subplot(2, 2, 3)
test_learning_rate(X, y, MiniBatchGD, learning_rates, max_iter=max_iter, stop_condition=StopCondition.WorseThanWorst, max_iter_no_imp=100)

max_iter = 20
epsilons = [1/10**i for i in range(1, 6)]
plt.subplot(2, 2, 4)
test_IRLS_regularization(X, y, epsilons, max_iter=max_iter, stop_condition=StopCondition.WorseThanWorst, max_iter_no_imp=100)
plt.show()
