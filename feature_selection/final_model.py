from feature_selection.dataset import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

dataset_number = 1
X = datasets[dataset_number].X_train
ss = StandardScaler().fit(X)
X = ss.transform(X)
y = datasets[dataset_number].y_train
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=123, test_size=0.2)
X_real_test = ss.transform(datasets[dataset_number].X_valid)

# features = [64, 281, 318, 378, 472, 475]
features = [443, 904, 1656, 3304, 3656, 4386]


results = []
for i in range(100):
    rf = RandomForestClassifier().fit(X_train[:, features], y_train)
    results.append(np.sum(rf.predict(X_test[:, features]) == y_test) / y_test.shape[0])

print(np.mean(results))

np.random.seed(123)
rf = RandomForestClassifier().fit(X[:, features], y)
preds = rf.predict_proba(X_real_test[:, features])[:, 1]
result_df = pd.DataFrame(preds, columns=["TOMMAK"])
# dataset = 'artificial'
dataset = 'digits'
result_df.to_csv(f'TOMMAK_{dataset}_prediction.txt', index=False)
pd.DataFrame(features, columns=["TOMMAK"]).to_csv(f'TOMMAK_{dataset}_features.txt', index=False)
