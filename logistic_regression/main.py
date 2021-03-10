from classifier import IRLS
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
X = np.array(iris.data)
y = np.array(iris.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

irls = IRLS()
irls.train(X_train, y_train)
-
