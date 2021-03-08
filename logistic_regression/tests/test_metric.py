import pytest
import numpy as np
from sklearn.utils._testing import assert_array_equal

from ..metric import accuracy, precision, recall, f1score


@pytest.fixture()
def y_true():
    return np.array([0, 1, 1, 0, 0])


def accuracy_test(y_true, y_pred, acc):
    assert accuracy(y_true, y_pred) == acc


def precision_test(y_true, y_pred, prec):
    assert precision(y_true, y_pred) == prec


def recall_test(y_true, y_pred, rec):
    assert recall(y_true, y_pred) == rec


def f1score_test(y_true, y_pred, f1):
    print(f1)
    assert f1score(y_true, y_pred) == f1


@pytest.mark.parametrize('y_pred, acc, prec, rec',
                         [(np.array([1, 1, 0, 0, 1]), 0.4, 1/3, 1/2),
                          (np.zeros(5), 0.6, 0, 0),
                          (np.ones(5), 0.4, 0.4, 1)])
def test_metrics(y_true, y_pred, acc, prec, rec):
    accuracy_test(y_true, y_pred, acc)
    precision_test(y_true, y_pred, prec)
    recall_test(y_true, y_pred, rec)
    if prec + rec == 0:
        f1score_test(y_true, y_pred, 2 * prec * rec / 10**-9)
    else:
        f1score_test(y_true, y_pred, 2 * prec * rec / (prec + rec))
