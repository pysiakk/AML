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
import matplotlib.pyplot as plt
import time

X = datasets[0].X_train
ss = StandardScaler().fit(X)
X = ss.transform(X)
y = datasets[0].y_train
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=123, test_size=0.2)


################################################################################
# first part
################################################################################

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
# df.to_csv('~/Desktop/AML_result1.csv')

df = pd.read_csv('~/Desktop/AML_result1.csv', index_col=0)
df.max(axis=0)
print(df[df['RandomForestClassifier'] >= 0.8].to_string())

columns = ['n_trees', 'max_iter', 'depth_factor', 'selection']
df_index = pd.DataFrame(df.index.str.split(', ').tolist(), index=df.index, columns=columns)
df = df.join(df_index)

df_tmp = pd.DataFrame(df.iloc[:, 2:12], dtype=float)
df_tmp['gt_acc'] = pd.DataFrame(df.iloc[:, 1:2], dtype=float)
df_tmp['selection'] = df.iloc[:, 12:13]
df_tmp.columns = ['Ada', 'GB', 'DT', 'RF', 'QDA', 'LDA', 'LR'] + df_tmp.columns[7:10].tolist() + ['GT'] + df_tmp.columns[11:12].tolist()


def save_plot(col_name):
    # col_name = columns[3]
    plt.rc('font', size=20)
    # a = df_tmp.groupby(col_name).mean().T.iloc[[3, 10]]
    # a.columns = ['Roulette', 'StochasticUniform', 'Tournament']
    # ax = a.plot(kind='bar', ylabel='accuracy', xlabel='classifier')
    ax = df_tmp.groupby(col_name).mean().T.iloc[[3, 10]].plot(kind='bar', ylabel='accuracy', xlabel='classifier')
    ax.legend(loc='lower right')
    plt.title(f'{col_name}')
    plt.tight_layout()
    plt.savefig(f'experiment1-{col_name}.png', pad_inches=0.2)


[save_plot(col) for col in columns]


################################################################################
# second part
################################################################################

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

df = pd.DataFrame(all_results).T
df.iloc[:, 3:].max(axis=0)
df.iloc[:, 1:2].max(axis=0)
print(df[df['RandomForestClassifier'] >= 0.85].to_string())
# df.to_csv('~/Desktop/AML_result2.csv')
# df = pd.read_csv('~/Desktop/AML_result2.csv', index_col=0)

columns = ['n_trees', 'n_thresholds', 'depth_factor', 'tournament_size']
df_index = pd.DataFrame(df.index.str.split(', ').tolist(), index=df.index, columns=columns)
df = df.join(df_index)

df_tmp = pd.DataFrame(df.iloc[:, 3:], dtype=float)
df_tmp['gt_acc'] = pd.DataFrame(df.iloc[:, 1:2], dtype=float)
df_tmp.columns = df_tmp.columns[0:1].tolist() + ['Ada', 'GB', 'DT', 'RF', 'QDA', 'LDA', 'LR'] + df_tmp.columns[8:12].tolist() + ['GT']


def save_plot(col_name):
    plt.rc('font', size=20)
    ax = df_tmp.groupby(col_name).mean().T.iloc[[4, 11]].plot(kind='bar', ylabel='accuracy', xlabel='classifier')
    ax.legend(loc='lower right')
    plt.title(f'{col_name}')
    plt.tight_layout()
    plt.savefig(f'experiment2-{col_name}.png', pad_inches=0.2)


[save_plot(col) for col in columns]


def get_features_importance(df, n_experiment):
    all_features = []
    features_weights = []
    for row, acc in zip(df['features'], df['RandomForestClassifier']):
        all_features += row.tolist()
        features_weights += [acc-0.5] * len(row)

    features_occurrences = {}
    for item in np.unique(all_features):
        features_occurrences[item] = np.sum(np.array((all_features) == item))

    features_df = pd.DataFrame(features_occurrences, index=['occurrences']).T
    features_df = features_df.sort_values(by='occurrences', ascending=False)

    results_rf = []
    for i in range(1, 21):
        rf_features = features_df.index[:i].tolist()
        rf = RandomForestClassifier().fit(X_train[:, rf_features], y_train)
        results_rf.append(np.sum(rf.predict(X_test[:, rf_features]) == y_test) / y_test.shape[0])

    features_occurrences2 = {}
    for item in np.unique(all_features):
        features_occurrences2[item] = np.sum(np.array(features_weights)[all_features == item])

    features_df2 = pd.DataFrame(features_occurrences2, index=['occurrences']).T
    features_df2 = features_df2.sort_values(by='occurrences', ascending=False)

    results_rf2 = []
    for i in range(1, 21):
        rf_features = features_df2.index[:i].tolist()
        rf = RandomForestClassifier().fit(X_train[:, rf_features], y_train)
        results_rf2.append(np.sum(rf.predict(X_test[:, rf_features]) == y_test) / y_test.shape[0])

    print(results_rf)
    print(results_rf2)

    plt.rc('font', size=20)
    plt.plot(results_rf, label='normal')
    plt.plot(results_rf2, label='weighted')
    plt.legend()
    plt.xlabel('number of features')
    plt.ylabel('accuracy')
    plt.title(f'Experiment{n_experiment}')
    plt.tight_layout()
    return features_df, features_df2


features_df, features_df2 = get_features_importance(df, 2)
plt.savefig(f'experiment2-rf_top_features.png', pad_inches=0.2)

features_df.index[:7].tolist()
# [241, 475, 338, 105, 336, 48, 451]
features_df2.index[:7].tolist()
# [241, 475, 105, 48, 338, 336, 451]

################################################################################
# third part
################################################################################

all_results = {}
max_iter = 800
selection = Selection.Tournament
n_trees = 500
n_thresholds = 100
depth_factor = 0.01
tournament_size = 5
for mutation_prob in [round(i*0.1, 2) for i in range(1, 11)]:
    cross_prob = round((1.6 - mutation_prob) / 2, 2)
    for n_elitism in [3, 5, 10]:
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

df = pd.DataFrame(all_results).T
df.iloc[:, 3:].max(axis=0)
df.iloc[:, 1:2].max(axis=0)
df.columns
df_max7 = df[df['n_features'] <= 7]
print(df_max7[df_max7['RandomForestClassifier'] >= 0.85].to_string())
# df.to_csv('~/Desktop/AML_result3.csv')
# df = pd.read_csv('~/Desktop/AML_result3.csv', index_col=0)

columns = ['mutation_prob', 'initialization', 'n_elitism']
df_index = pd.DataFrame(df.index.str.split(', ').tolist(), index=df.index, columns=columns)
df = df.join(df_index)

df_tmp = pd.DataFrame(df.iloc[:, 3:11], dtype=float)
df_tmp['gt_acc'] = pd.DataFrame(df.iloc[:, 1:2], dtype=float)
for i in range(3):
    df_tmp[columns[i]] = pd.DataFrame(df.iloc[:, 11+i])
df_tmp.columns = df_tmp.columns[0:1].tolist() + ['Ada', 'GB', 'DT', 'RF', 'QDA', 'LDA', 'LR', 'GT'] + df_tmp.columns[9:12].tolist()


def save_plot(col_name):
    # col_name = columns[0]
    plt.rc('font', size=20)
    ax = df_tmp.groupby(col_name).mean().T.iloc[[4, 8]].plot(kind='bar', ylabel='accuracy', xlabel='classifier')
    # ax.legend(loc='lower right', ncol=3)
    ax.legend(loc='lower right')
    plt.title(f'{col_name}')
    plt.tight_layout()
    plt.savefig(f'experiment3-{col_name}.png', pad_inches=0.2)


[save_plot(col) for col in columns]

features_df, features_df2 = get_features_importance(df, 3)
plt.savefig(f'experiment3-rf_top_features.png', pad_inches=0.2)

features_df.index[:7].tolist()
# [475, 128, 338, 318, 48, 378, 241]
features_df2.index[:7].tolist()
# [475, 128, 318, 338, 48, 241, 378]
features_df.index[:4].tolist()
# [475, 128, 338, 318]
features_df2.index[:4].tolist()
# [475, 128, 318, 338]


################################################################################
# fourth experiment - for dataset digits
################################################################################


X = datasets[1].X_train
ss = StandardScaler().fit(X)
X = ss.transform(X)
y = datasets[1].y_train
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=123, test_size=0.2)


all_results = {}
max_iter = 2000
selection = Selection.Tournament
n_trees = 500
n_thresholds = 10
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


df = pd.DataFrame(all_results).T
plt.rc('font', size=20)
df_tmp = pd.DataFrame(df.iloc[:, 4:11], dtype=float)
df_tmp['gt_acc'] = pd.DataFrame(df.iloc[:, 1:2], dtype=float)
df_tmp.columns = ['Ada', 'GB', 'DT', 'RF', 'QDA', 'LDA', 'LR', 'GT']

print(df_tmp.to_string())
df.iloc[:, 3:4]
ax = df_tmp.T.plot(kind='bar', ylabel='accuracy', xlabel='classifier')
# ax.legend(loc='lower right', ncol=3)
ax.legend(loc='lower right')
plt.title(f'')
plt.tight_layout()
plt.savefig(f'digits.png', pad_inches=0.2)
