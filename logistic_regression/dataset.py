import pandas as pd
import numpy as np


class Dataset:
    def __init__(self, name):
        self.name = name
        train = pd.read_csv(f'logistic_regression/data/{name}_train.csv', index_col=0)
        self.X_train = np.array(train.iloc[:, :-1])
        self.y_train = self._change_class_to_0_1(train.iloc[:, -1])
        test = pd.read_csv(f'logistic_regression/data/{name}_test.csv', index_col=0)
        self.X_test = np.array(test.iloc[:, :-1])
        self.y_test = self._change_class_to_0_1(test.iloc[:, -1])

    @staticmethod
    def _change_class_to_0_1(df):
        array = np.array(df)
        if np.sum(array==2) > 0:
            array[array==1] = 0
            array[array==2] = 1
        return array


names = ['banknote', 'diabetes', 'madelon', 'ozone', 'steel']
datasets = []
for name in names:
    datasets.append(Dataset(name))

