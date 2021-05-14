import pandas as pd
import numpy as np


class Dataset:
    def __init__(self, name):
        self.name = name
        self.X_train = np.array(pd.read_csv("feature_selection/data/" + name + "_train.data", header=None, sep=" ").iloc[:, :-1])
        self.y_train = (np.array(pd.read_csv("feature_selection/data/" + name + "_train.labels", header=None, sep=" ")) + 1 ) // 2
        self.X_valid = np.array(pd.read_csv("feature_selection/data/" + name + "_valid.data", header=None, sep=" ").iloc[:, :-1])


names = ['artificial', "digits"]
datasets = []
for name in names:
    datasets.append(Dataset(name))

