from numpy import array
from hyperPipes import HyperPipes
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = datasets.load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

classifier = HyperPipes()
y_predicted = classifier.fit(X_train, y_train).predict(X_test)

accuracy = accuracy_score(y_test, y_predicted)

print accuracy