from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors="sqrt", weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.knn_ = None
        self.classes_ = [-1, 1]

    def fit(self, X, y):
        if self.n_neighbors == "sqrt":
            n = int(np.sqrt(len(X)))
        elif isinstance(self.n_neighbors, float):
            n = int(self.n_neighbors * len(X))
        else:
            n = self.n_neighbors
        self.knn_ = KNeighborsClassifier(n_neighbors=n, weights=self.weights).fit(X, y)

        return self

    def predict(self, X):
        return self.knn_.predict(X)

    def predict_proba(self, X):
        return self.knn_.predict_proba(X)