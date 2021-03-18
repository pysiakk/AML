from logistic_regression.classifier import IRLS, SGD, GD, MiniBatchGD
from logistic_regression.metric import Metric
import matplotlib.pyplot as plt
from logistic_regression.stopper import StopCondition
from logistic_regression.dataset import datasets

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

kwargs = {'stop_condition': StopCondition.WorseThanWorst,
          'max_iter_no_imp': 100}
models = [IRLS, SGD, GD, MiniBatchGD, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, KNeighborsClassifier]
metrics = [Metric.Acc, Metric.Precision, Metric.Recall, Metric.F1score]

scores = []
for dataset in datasets:
    for model_number, model in enumerate(models):
        if model_number < 4:
            m = model(**kwargs)
        else:
            m = model()
        m.fit(dataset.X_train, dataset.y_train)
        for metric in metrics:
            if model_number < 4:
                score = m.score(dataset.X_test, dataset.y_test, metric)
            else:
                score = metric.evaluate(dataset.y_test, m.predict(dataset.X_test))
            scores.append((dataset.name, model.__name__, metric.name, score))

print(scores)
print('Accuracies: ')
for score in scores[::4]:
    print(f'{score[3]:0.03f}', score[0], score[1])
