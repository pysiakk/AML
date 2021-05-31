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

correlation = [48, 128, 336, 338, 472, 475]
rf_importance = [105, 241, 338, 475]
rfe = [105, 318, 338, 378, 442, 475]
experiment2 = [241, 475, 338, 105, 336, 48, 451]
experiment3 = [475, 128, 338, 318]
experiment3_best = [105, 281, 338, 378, 433, 472, 475]

df = pd.DataFrame({
    'experiment3_best': {item: True for item in experiment3_best},
    'experiment3': {item: True for item in experiment3},
    'experiment2': {item: True for item in experiment2},
    'correlation': {item: True for item in correlation},
    'rfe': {item: True for item in rfe},
    'rf_importance': {item: True for item in rf_importance},
})

agg = df.sum(axis=1).sort_values(ascending=False)

results_rf = []
for i in range(1, 8):
    rf_features = agg.index[:i].tolist()
    rf = RandomForestClassifier().fit(X_train[:, rf_features], y_train)
    results_rf.append(np.sum(rf.predict(X_test[:, rf_features]) == y_test) / y_test.shape[0])


all_features = agg.index.tolist()
X_train = X_train[:, all_features]
X_test = X_test[:, all_features]

all_results = {}
max_iter = 800
selection = Selection.Tournament
n_trees = 500
n_thresholds = 100
depth_factor = 0.01
tournament_size = 5
for mutation_prob in [0.4, 0.9]:
    cross_prob = round((1.6 - mutation_prob) / 2, 2)
    for n_elitism in [5, 10]:
        results = {}
        start = time.time()
        gt = GeneticTree(n_trees=n_trees, max_iter=max_iter, n_thresholds=n_thresholds,
                         metric=Metric.AccuracyMinusDepth, depth_factor=depth_factor, selection=selection,
                         tournament_size=tournament_size, cross_prob=cross_prob, mutation_prob=mutation_prob,
                         random_state=123, n_elitism=n_elitism).fit(X_train, y_train)
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
        name = f'{mutation_prob}, {n_elitism}'
        print(name, '/n', results, '/n')
        all_results[name] = results
        pd.DataFrame(all_results).T.to_csv(f'~/Desktop/tmp/{name}')
