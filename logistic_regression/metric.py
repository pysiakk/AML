import numpy as np
from aenum import Enum, extend_enum
from sklearn.metrics import r2_score


def accuracy(labels_true, pred_proba, **kwargs):
    labels_pred = np.round(pred_proba)
    return np.sum(labels_true == labels_pred) / labels_true.shape[0]


def true_positive(labels_true, pred_proba, **kwargs):
    labels_pred = np.round(pred_proba)
    return np.sum(np.logical_and(labels_true == labels_pred, labels_true == 1))


def predicted_positive(labels_true, pred_proba, **kwargs):
    labels_pred = np.round(pred_proba)
    return np.sum(labels_pred == 1)


def condition_positive(labels_true, pred_proba, **kwargs):
    labels_pred = np.round(pred_proba)
    return np.sum(labels_true == 1)


def precision(labels_true, pred_proba, **kwargs):
    labels_pred = np.round(pred_proba)
    pp = predicted_positive(labels_true, labels_pred)
    if pp == 0:
        pp = 10**-9
    return true_positive(labels_true, labels_pred) / pp


def recall(labels_true, pred_proba, **kwargs):
    labels_pred = np.round(pred_proba)
    cp = condition_positive(labels_true, labels_pred)
    if cp == 0:
        cp = 10**-9
    return true_positive(labels_true, labels_pred) / cp


def f1score(labels_true, pred_proba, **kwargs):
    labels_pred = np.round(pred_proba)
    prec = precision(labels_true, labels_pred)
    rec = recall(labels_true, labels_pred)
    if prec + rec == 0:
        return 2*prec*rec / 10**-9
    else:
        return 2*prec*rec / (prec + rec)


def log_likelihood(labels_true, pred_proba, **kwargs):
    labels_true = labels_true.reshape((-1, 1))
    pred_proba = pred_proba.reshape((-1, 1))
    eps = 10**-12
    pred_proba[pred_proba <= 0] = eps
    pred_proba[pred_proba >= 1] = 1-eps
    ll = (np.log(pred_proba).T @ labels_true + np.log(1 - pred_proba).T @ (1 - labels_true))[0, 0]
    return ll


def r2(labels_true, pred_proba, classifier):
    # model = classifier.__class__(intercept=False, stop_condition=classifier.stop_condition, **classifier.kwargs)
    # model.fit(np.ones((len(labels_true), 1)), labels_true)
    # y_pred_proba_null = model.predict_proba(np.ones((len(labels_true), 1)))
    # return 1 - (log_likelihood(labels_true, pred_proba)/log_likelihood(labels_true, y_pred_proba_null))
    # labels_pred = np.round(pred_proba)
    return r2_score(labels_true, pred_proba)


class Metric(Enum):
    def __new__(cls, function, *args):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__)
        obj.evaluate = function
        return obj

    @staticmethod
    def add_new(name, function):
        extend_enum(Metric, name, function)

    Acc = accuracy,
    Precision = precision,
    Recall = recall,
    F1score = f1score,
    LogLikelihood = log_likelihood,
    R2 = r2,
