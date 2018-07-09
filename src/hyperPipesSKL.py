from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import numpy as np


class TemplateClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.n_dimensions = 0
        self.numerical_bounds = []

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.y_unique_values, self.y_unique_indices = np.unique(
            y, return_inverse=True)
        self.n_y_unique = self.y_unique_values.shape[0]
        self.hyper_pipes = [HyperPipe() for i in range(self.n_y_unique)]

        for i in range(self.n_y_unique):
            target_class = self.y_unique_values[i]
            target_class_indices = np.where(y == target_class)
            data_x_filtered = X[target_class_indices]
            self.hyper_pipes[i].fit(data_x_filtered, target_class)

        return self

    def predict(self, X):

        #scores = []
        predictions = []
        for instance in X:
            partial_results = []
            for i in range(self.n_y_unique):
                partial_results.append(
                    self.hyper_pipes[i].partial_contains(instance))
            best = max(partial_results, key=lambda item: item[0])[1]
            predictions.append(best)

        return predictions

class HyperPipe:

    def __init__(self):
        self.n_dimensions = 0
        self.numerical_bounds = []

    def fit(self, data_x, target_class):
        self.target_class = target_class
        self.n_dimensions = data_x.shape[1]

        # Initializes bounds
        for i in range(self.n_dimensions):
            bounds = []
            bounds.append(float('+inf'))  # lower bound
            bounds.append(float('-inf'))  # upper bound
            self.numerical_bounds.append(bounds)

        # Add instances
        for i in range(data_x.shape[0]):
            self.__add_instance__(data_x[i])

        return None

    def __add_instance__(self, data_x):
        #check boundaries
        for i in range(self.n_dimensions):
            if(data_x[i] < self.numerical_bounds[i][0]):
                self.numerical_bounds[i][0] = data_x[i]
            if(data_x[i] > self.numerical_bounds[i][1]):
                self.numerical_bounds[i][1] = data_x[i]

        return None

    def partial_contains(self, data_x):
        count = 0
        for i in range(self.n_dimensions):
            if(data_x[i] > self.numerical_bounds[i][0] and data_x[i] < self.numerical_bounds[i][1]):
                count += 1
        score = float(count) / self.n_dimensions

        return (score, self.target_class)
