from logistic_regression.stopper import Stopper
from logistic_regression.metric import Metric
import numpy as np


class Classifier:
    def __init__(self, **kwargs):
        self.stopper = Stopper(**kwargs)
        self.beta = None

    def train(self, X, y):
        y = y.reshape(-1, 1)
        self.beta = np.zeros([X.shape[1], 1])
        while not self.stopper.stop():
            self._train_iteration(X, y)
        return self

    def predict_proba(self, X):
        return self._predict(X).reshape(-1)

    def predict(self, X, threshold=0.5):
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


class IRLS(Classifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_derivative(self, X, y, p):
        # TODO regularization on diagonal in W
        W = np.diag([x*(1-x) for x in p.reshape(-1)])
        return np.linalg.inv(X.transpose() @ W @ X) @ X.transpose() @ (y.reshape((-1, 1)) - p.reshape(-1, 1))


class GeneralGradientDescent(Classifier):

    def __init__(self, batch_size: int, learning_rate: float = 0.01, **kwargs):
        super(GeneralGradientDescent, self).__init__(**kwargs)
        self.batch_size: int = batch_size
        self.learning_rate: float = 0.01

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
        super().__init__(-1, **kwargs)


class SGD(GeneralGradientDescent):

    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)


class MiniBatchGD(GeneralGradientDescent):

    def __init__(self, **kwargs):
        super().__init__(32, **kwargs)

