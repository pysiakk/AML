import numpy as np
from aenum import Enum, extend_enum


def accuracy(labels_true, labels_pred):
    return np.sum(labels_true == labels_pred) / labels_true.shape[0]


def true_positive(labels_true, labels_pred):
    return np.sum(np.logical_and(labels_true == labels_pred, labels_true == 1))


def predicted_positive(labels_true, labels_pred):
    return np.sum(labels_pred == 1)


def condition_positive(labels_true, labels_pred):
    return np.sum(labels_true == 1)


def precision(labels_true, labels_pred):
    pp = predicted_positive(labels_true, labels_pred)
    if pp == 0:
        pp = 10**-9
    return true_positive(labels_true, labels_pred) / pp


def recall(labels_true, labels_pred):
    cp = condition_positive(labels_true, labels_pred)
    if cp == 0:
        cp = 10**-9
    return true_positive(labels_true, labels_pred) / cp


def f1score(labels_true, labels_pred):
    prec = precision(labels_true, labels_pred)
    rec = recall(labels_true, labels_pred)
    if prec + rec == 0:
        return 2*prec*rec / 10**-9
    else:
        return 2*prec*rec / (prec + rec)


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
