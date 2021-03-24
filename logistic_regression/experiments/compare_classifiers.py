from logistic_regression.classifier import IRLS, SGD, GD, MiniBatchGD, LDA, QDA, KNN
from logistic_regression.metric import Metric
import matplotlib.pyplot as plt
from logistic_regression.stopper import StopCondition
from logistic_regression.dataset import datasets
import numpy as np

kwargs = {'stop_condition': StopCondition.WorseThanWorst,
          'max_iter_no_imp': 100}
models = [IRLS, SGD, GD, MiniBatchGD, LDA, QDA, KNN]
metrics = [Metric.Acc, Metric.Precision, Metric.Recall, Metric.F1score, Metric.R2]

plt.subplots(3, 2, figsize=(9.6, 10.8))
for dataset_number, dataset in enumerate(datasets):
    plt.subplot(3, 2, dataset_number+1)
    for model_number, model in enumerate(models):
        m = model(**kwargs)
        m.fit(dataset.X_train, dataset.y_train)
        scores = []
        for metric_number, metric in enumerate(metrics):
            score = m.score(dataset.X_test, dataset.y_test, metric)
            scores.append(score)
        plt.bar(np.arange(1, len(metrics)+1) + 0.1*(model_number-3), scores, width=0.1, label=model.__name__)
    plt.xticks(np.arange(1, len(metrics) + 1), [metric.name for metric in metrics])
    plt.legend(loc=3)
    plt.title(label=f'Comparison of classifiers at {dataset.name}')
plt.show()
