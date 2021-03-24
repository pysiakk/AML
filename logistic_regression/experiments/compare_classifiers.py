from logistic_regression.classifier import IRLS, SGD, GD, MiniBatchGD, LDA, QDA, KNN
from logistic_regression.metric import Metric
import matplotlib.pyplot as plt
from logistic_regression.stopper import StopCondition
from logistic_regression.dataset import datasets
import numpy as np
import scipy as sc

kwargs = {'stop_condition': StopCondition.WorseThanWorst,
          'max_iter_no_imp': 100}
models = [IRLS, SGD, GD, MiniBatchGD, LDA, QDA, KNN]
metrics = [Metric.Acc, Metric.Precision, Metric.Recall, Metric.F1score, Metric.R2]

for dataset in datasets:
    addition = -0.4
    for model in models:
        m = model(**kwargs)
        m.fit(dataset.X_train, dataset.y_train)
        scores = []
        for metric_number, metric in enumerate(metrics):
            score = m.score(dataset.X_test, dataset.y_test, metric)
            scores.append(score)
        addition += 0.1
        plt.bar(np.arange(1, len(metrics)+1) + addition, scores, width=0.1, label=model.__name__)
    plt.xticks(np.arange(1, len(metrics) + 1), [metric.name for metric in metrics])
    plt.legend(loc=3)
    plt.title(label=f'Comparison of classifiers at {dataset.name}')
    plt.show()
