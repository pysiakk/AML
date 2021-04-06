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

plt.subplots(3, 2, figsize=(12, 9))
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(label=f'Comparison of classifiers at {dataset.name}')
plt.show()
