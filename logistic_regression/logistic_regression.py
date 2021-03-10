from .stopper import Stopper
import numpy as np


class Classifier:
    def __init__(self, **kwargs):
        self.stopper = Stopper(**kwargs)
        self.beta = None

    def train(self, X, y):
        self.beta = np.ones([X.shape[1], 1])
        while not self.stopper.stop():
            self._train_iteration(X, y)

    def _train_iteration(self, X, y):
        self.beta += self._compute_derivative(X, y)
        self.p = self._predict(X)

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
        W = np.diag([x*(1-x) for x in p])
        return np.linalg.inv(X.transpose() @ W @ X) @ X.transpose() @ (y - p)



class GeneralGradientDescent(Classifier):

    def __init__(self, batch_size: int, **kwargs):
        super(GeneralGradientDescent, self).__init__(**kwargs)
        self.batch_size: int = batch_size


class GD(GeneralGradientDescent):

    def __init__(self, **kwargs):
        super().__init__(-1, **kwargs)


class SGD(GeneralGradientDescent):

    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)


class MiniBatchGD(GeneralGradientDescent):

    def __init__(self, **kwargs):
        super().__init__(32, **kwargs)

