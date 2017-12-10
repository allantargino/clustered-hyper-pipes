from numpy import array
from hyperPipes import HyperPipes
from sklearn import datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X, y = iris.data, iris.target

classifier = HyperPipes()
y_predicted = classifier.fit(X, y).predict(X)

accuracy = accuracy_score(y, y_predicted)

print accuracy