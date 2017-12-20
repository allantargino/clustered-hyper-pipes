import numpy as np
import os

class custom_datasets():
    
    @classmethod
    def load_ionosphere(self):
        return self.load_custom_dataset('..\\datasets\\ionosphere.data')

    @classmethod
    def load_wine(self):
        return self.load_custom_dataset('..\\datasets\\wine_processed.data')

    @classmethod
    def load_live_disorders(self):
        return self.load_custom_dataset('..\\datasets\\bupa.data')

    @classmethod
    def load_custom_dataset(self, path):
        dir = os.path.dirname(__file__)
        file_name = os.path.join(dir, path)

        data = np.genfromtxt(file_name, delimiter=',')
        n_cols = data.shape[1]

        X = data[:,0:n_cols-1]
        y =np.genfromtxt(file_name, delimiter=',', usecols=(n_cols-1), dtype=None)

        return Dataset(X, y)

class Dataset:

    def __init__(self, X, y):
        self.data = X
        self.target = y