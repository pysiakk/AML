from feature_selection.dataset import datasets
import pandas as pd
import numpy as np


class SelectedFeatures:
    def __init__(self, dataset_name, features, method):
        self.dataset_name = dataset_name
        self.features = features
        self.method = method


def target_correlation(dataset, threshold):
    df = pd.DataFrame(np.hstack((dataset.X_train, dataset.y_train)))
    corr_matrix = np.abs(df.corr())
    corr = [[i, corr_matrix[corr_matrix.columns[-1]][i]] for i in corr_matrix[corr_matrix.columns[-1]][:-1].index]
    corr = sorted(corr, key=lambda x: -x[1])
    features = [corr[0][0]]
    for i, row in enumerate(corr):
        if np.max(corr_matrix[row[0]][features]) < .9 and row[1] > threshold:
            features.append(row[0])
    return features


print(target_correlation(datasets[0], 0.1))
