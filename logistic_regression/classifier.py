from logistic_regression.stopper import Stopper
from logistic_regression.metric import Metric
import numpy as np


class Classifier:
    def __init__(self, intercept=True, stop_condition=None, **kwargs):
        self.stopper = Stopper(stop_condition=stop_condition, **kwargs)
        self.beta = None
        self.intercept = intercept
        self.log_likelihood = []

    def train(self, X, y):
        if self.intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = y.reshape(-1, 1)
        self.beta = np.zeros([X.shape[1], 1])
        while not self.stopper.stop(self):
            self._train_iteration(X, y)
            y_pred_proba = self._predict(X)
            self.log_likelihood.append(self._log_likelihood(y, y_pred_proba))
        return self

    def predict_proba(self, X):
        if self.intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return self._predict(X).reshape(-1)

    def predict(self, X, threshold=0.5):
        if self.intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_pred = np.zeros(X.shape[0])
        y_pred[self._predict(X).reshape(-1) > threshold] = 1
        return y_pred

    def score(self, X, y_true, metric: Metric):
        y_pred = self.predict(X)
        return metric.evaluate(y_true, y_pred)

    def _train_iteration(self, X, y):
        p = self._predict(X)
        self.beta += self._compute_derivative(X, y, p)

    def _compute_derivative(self, X, y, p):
        pass

    def _predict(self, X):
        """
        :param X: matrix with observations: n_observations x n_predictors
        :return: predictions as np.array n_observations x 1
        """
        return 1 / (1 + np.exp(-X @ self.beta))

    @staticmethod
    def _log_likelihood(y_true, y_pred_proba):
        return (np.log(y_pred_proba).T @ y_true + np.log(1 - y_pred_proba).T @ (1 - y_true))[0, 0]


class IRLS(Classifier):

    def __init__(self, eps=0, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def _compute_derivative(self, X, y, p):
        W = np.diag([x*(1-x) + self.eps for x in p.reshape(-1)])
        return np.linalg.inv(X.T @ W @ X) @ X.T @ (y.reshape((-1, 1)) - p.reshape(-1, 1))


class GeneralGradientDescent(Classifier):

    def __init__(self, batch_size: int, learning_rate: float = 0.01, **kwargs):
        super(GeneralGradientDescent, self).__init__(**kwargs)
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate

    @staticmethod
    def _shuffle(X, y, p):
        s = np.arange(X.shape[0])
        np.random.shuffle(s)
        return X[s, :], y[s], p[s]

    def _get_batch_data(self, X, y, p):
        X = X[:self.batch_size, :]
        y = y[:self.batch_size]
        p = p[:self.batch_size]
        return X, y, p

    def _compute_derivative(self, X, y, p):
        if self.batch_size != -1:
            X, y, p = self._shuffle(X, y, p)
            X, y, p = self._get_batch_data(X, y, p)
        return - self.learning_rate / X.shape[0] * (X.T @ (p - y))


class GD(GeneralGradientDescent):

    def __init__(self, **kwargs):
        super().__init__(batch_size=-1, **kwargs)


class SGD(GeneralGradientDescent):

    def __init__(self, **kwargs):
        super().__init__(batch_size=1, **kwargs)


class MiniBatchGD(GeneralGradientDescent):

    def __init__(self, **kwargs):
        super().__init__(batch_size=32, **kwargs)

