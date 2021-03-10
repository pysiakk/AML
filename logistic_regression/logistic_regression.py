from .stopper import Stopper
import numpy as np


class Classifier:
    def __init__(self, **kwargs):
        self.stopper = Stopper(**kwargs)
        self.beta = None

    def train(self, X, y):
        self.beta = np.ones(X.shape[1])
        while not self.stopper.stop():
            self._train_iteration()

    def _train_iteration(self):
        self.beta += self._compute_derivative()

    def _compute_derivative(self):
        pass


class IRLS(Classifier):
    pass


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

