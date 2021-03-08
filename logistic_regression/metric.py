import numpy as np
from aenum import Enum, extend_enum


def accuracy(labels_true, labels_pred):
    return np.sum(labels_true == labels_pred) / labels_true.shape[0]


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