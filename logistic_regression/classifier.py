from logistic_regression.stopper import Stopper
from logistic_regression.metric import Metric
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


class Classifier:
    def __init__(self, intercept=True, stop_condition=None, **kwargs):
        self.stop_condition = stop_condition
        self.stopper = Stopper(stop_condition=stop_condition, **kwargs)
        self.beta = None
        self.intercept = intercept
        self.log_likelihood = []
        self.kwargs = kwargs

    def fit(self, X, y):
        if self.intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = y.reshape(-1, 1)
        self.beta = np.zeros([X.shape[1], 1])
        while not self.stopper.stop(self):
            self._train_iteration(X, y)
            y_pred_proba = self._predict(X)
            self.log_likelihood.append(Metric.LogLikelihood.evaluate(y, y_pred_proba))
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
        y_pred_proba = self.predict_proba(X)
        return metric.evaluate(y_true, y_pred_proba, classifier=self)

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

    def __init__(self, eps=0.001, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def _compute_derivative(self, X, y, p):
        W = np.diag([x*(1-x) + self.eps for x in p.reshape(-1)])
        return np.linalg.pinv(X.T @ W @ X) @ X.T @ (y.reshape((-1, 1)) - p.reshape(-1, 1))


class GeneralGradientDescent(Classifier):

    def __init__(self, batch_size: int, learning_rate: float = 0.05, **kwargs):
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


class ImportedClassifier(Classifier):

    def __init__(self, base, classifier_kwargs, **kwargs):
        self.classifier = base(**classifier_kwargs)
        self.kwargs = kwargs
        self.stop_condition = None

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X, threshold=0.5):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)[:, 1]


class LDA(ImportedClassifier):
    def __init__(self, **kwargs):
        super().__init__(LinearDiscriminantAnalysis, {'solver': 'lsqr'})


class QDA(ImportedClassifier):
    def __init__(self, **kwargs):
        super().__init__(QuadraticDiscriminantAnalysis, {'reg_param': 0.001})


class KNN(ImportedClassifier):
    def __init__(self, **kwargs):
        super().__init__(KNeighborsClassifier, {})
