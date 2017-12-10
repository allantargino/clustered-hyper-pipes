import numpy as np

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


class HyperPipes:

    def __init__(self):
        self.hyper_pipes = []

    def fit(self, data_x, data_y):
        self.y_unique_values, self.y_unique_indices = np.unique(
            data_y, return_inverse=True)
        self.n_y_unique = self.y_unique_values.shape[0]
        self.hyper_pipes = [HyperPipe() for i in range(self.n_y_unique)]

        for i in range(self.n_y_unique):
            target_class = self.y_unique_values[i]
            target_class_indices = np.where(data_y == target_class)
            data_x_filtered = data_x[target_class_indices]
            self.hyper_pipes[i].fit(data_x_filtered, target_class)

        return self

    def predict(self, data_x):
        #scores = []
        predictions = []
        for instance in data_x:
            partial_results = []
            for i in range(self.n_y_unique):
                partial_results.append(self.hyper_pipes[i].partial_contains(instance))
            best = max(partial_results,key=lambda item:item[0])[1]
            predictions.append(best)

        return predictions
