from genetic_tree.genetic_tree import GeneticTree
from genetic_tree.genetic.evaluator import Metric
from genetic_tree.genetic.selector import Selection
from feature_selection.dataset import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
import time

X = datasets[0].X_train
ss = StandardScaler().fit(X)
X = ss.transform(X)
y = datasets[0].y_train
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=123, test_size=0.2)

all_results = {}
for n_trees in [100, 200, 500]:
    for max_iter in [200, 400, 800]:
        for depth_factor in [0.01, 0.001, 0.0001]:
            for selection in [Selection.StochasticUniform, Selection.Roulette, Selection.Tournament]:
                results = {}
                start = time.time()
                gt = GeneticTree(n_trees=n_trees, max_iter=max_iter, metric=Metric.AccuracyMinusDepth, depth_factor=depth_factor, selection=selection).fit(X_train, y_train)
                time_delta = time.time()-start
                results['time'] = time_delta
                acc = np.sum(gt.predict(X_test) == y_test) / y_test.shape
                results['gt_acc'] = acc[0]
                features = np.unique(gt._best_tree.feature[gt._best_tree.children_left != -1])

                classifiers = [AdaBoostClassifier, GradientBoostingClassifier, DecisionTreeClassifier, RandomForestClassifier,
                               QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis, LogisticRegression]
                for classifier in classifiers:
                    cl = classifier().fit(X_train[:, features], y_train)
                    results[classifier.__name__] = np.sum(cl.predict(X_test[:, features]) == y_test) / y_test.shape[0]
                print(results)
                name = f'{n_trees}, {max_iter}, {depth_factor}, {selection.select.__name__}'
                all_results[name] = results
                pd.DataFrame(all_results).T.to_csv(f'~/Desktop/tmp/{name}')

df = pd.DataFrame(all_results).T
df.to_csv('~/Desktop/AML_result1.csv')

df = pd.read_csv('~/Desktop/AML_result1.csv')
df.max(axis=0)
print(df[df['RandomForestClassifier'] >= 0.8].to_string())



all_results = {}
max_iter = 800
selection = Selection.Tournament
for n_trees in [300, 400, 500]:
    for n_thresholds in [10, 30, 100]:
        for depth_factor in [0.01, 0.003, 0.001]:
            for tournament_size in [2, 3, 5]:
                results = {}
                start = time.time()
                gt = GeneticTree(n_trees=n_trees, max_iter=max_iter, n_thresholds=n_thresholds, metric=Metric.AccuracyMinusDepth, depth_factor=depth_factor, selection=selection, tournament_size=tournament_size, random_state=123).fit(X_train, y_train)
                time_delta = time.time()-start
                results['time'] = time_delta
                acc = np.sum(gt.predict(X_test) == y_test) / y_test.shape
                results['gt_acc'] = acc[0]
                features = np.unique(gt._best_tree.feature[gt._best_tree.children_left != -1])
                results['features'] = features
                results['n_features'] = len(features)

                classifiers = [AdaBoostClassifier, GradientBoostingClassifier, DecisionTreeClassifier, RandomForestClassifier,
                               QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis, LogisticRegression]
                for classifier in classifiers:
                    cl = classifier().fit(X_train[:, features], y_train)
                    results[classifier.__name__] = np.sum(cl.predict(X_test[:, features]) == y_test) / y_test.shape[0]
                name = f'{n_trees}, {n_thresholds}, {depth_factor}, {tournament_size}'
                print(name, '/n', results, '/n')
                all_results[name] = results
                pd.DataFrame(all_results).T.to_csv(f'~/Desktop/tmp/{name}')
