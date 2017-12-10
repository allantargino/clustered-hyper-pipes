from numpy import array
from hyperPipes import HyperPipes
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

hp = HyperPipes()
y_predicted = hp.fit(X_train, y_train).predict(X_test)
accuracy = accuracy_score(y_test, y_predicted)
print 'HyperPipes:\t' + str(accuracy)

classifier = GaussianNB()
y_predicted = classifier.fit(X_train, y_train).predict(X_test)
accuracy = accuracy_score(y_test, y_predicted)
print 'GaussianNB:\t' + str(accuracy)