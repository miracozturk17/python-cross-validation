from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

rkf = RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)
model = KNeighborsClassifier()

accuracies = []

for train_index, test_index in rkf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    accuracies.append(accuracy)

print("Ortalama dogruluk:", sum(accuracies) / len(accuracies))