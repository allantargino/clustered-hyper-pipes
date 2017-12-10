import numpy as np

class ClassDecomposition:
    
    def __init__(self, algorithm, k):
        self.algorithm = algorithm
        self.k = k

    def decompose(self, data_x, data_y):
        self.y_unique_values, self.y_unique_indices = np.unique(
            data_y, return_inverse=True)
        self.n_y_unique = self.y_unique_values.shape[0]

        renamed_data_x = np.array([])
        renamed_data_y = np.array([])
        for i in range(self.n_y_unique):
            target_class = self.y_unique_values[i]
            target_class_indices = np.where(data_y == target_class)

            data_i_x = data_x[target_class_indices]
            data_i_y = data_y[target_class_indices]
            fitted = self.algorithm.fit(data_i_x)
            labels = fitted.labels_
            
            for j in range(self.k):
                indices_j = np.where(labels == j)
                data_j_x = data_i_x[indices_j]
                data_j_y = data_i_y[indices_j]

                relabed = self.relabel(data_j_y, j)

                renamed_data_x = np.append(renamed_data_x, data_j_x)
                renamed_data_y = np.append(renamed_data_y, relabed)

        X = np.reshape(renamed_data_x, (data_x.shape[0], data_x.shape[1]))
        y = renamed_data_y
        
        return X, y

    def relabel(self, data, cluster_idx):
        return np.array(map(lambda item: str(item) + chr(65 + cluster_idx), data))