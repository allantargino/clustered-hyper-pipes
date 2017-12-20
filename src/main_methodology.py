import numpy as np
from hyperPipes import HyperPipes
from classDecomposition import ClassDecomposition
from customDatasets import custom_datasets
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# Parameters:
classifiers_names = [
    "Naive Bayes"
]

classifiers = [
    GaussianNB()
]

k_values = [
    1,
    2,
    3,
    4,
    5,
    6,
    7
]

datasets_names = [
    "Breast",
    "Iris",
    "Ionosphere",
    "Wine",
    "Live Disorders"
]

datasets = [
    datasets.load_breast_cancer(),
    datasets.load_iris(),
    custom_datasets.load_ionosphere(),
    custom_datasets.load_wine(),
    custom_datasets.load_live_disorders()
]

# iterate over ks
for k in k_values:
    print "k\t" + str(k)

    # iterate over datasets
    for ds_name, ds in zip(datasets_names, datasets):
        print "\tDataset\t" + ds_name
        # preprocess dataset, split into training and test part
        X, y = ds.data, ds.target
        X = StandardScaler().fit_transform(X)

        cd = ClassDecomposition(KMeans(n_clusters=k), k)
        X, y = cd.decompose(X, y)

        folds = StratifiedKFold(10)

        # iterate over classifiers
        for clf_name, clf in zip(classifiers_names, classifiers):
            accuracy = cross_val_score(clf, X, y, cv=folds, scoring='accuracy')
            m = np.mean(accuracy)
            a = np.std(accuracy)
            print "\t\t" + clf_name + "\t" + str(m) + '土' + str(a)
