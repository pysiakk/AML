import pytest
import numpy as np
from sklearn.utils._testing import assert_array_equal

from ..metric import accuracy


@pytest.fixture()
def y_true():
    return np.array([0, 1, 1, 0, 0])


@pytest.mark.parametrize('y_pred, acc',
                         [(np.array([1, 1, 0, 0, 1]), 0.4),
                          (np.zeros(5), 0.6),
                          (np.ones(5), 0.4)])
def test_accuracy(y_true, y_pred, acc):
    assert accuracy(y_true, y_pred) == acc
