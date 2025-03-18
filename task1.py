import numpy as np
import pandas as pd

def kNN_Classifier():
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def euclidean_distance(self, X1, X2):
        for a, b in zip(X1, X2):
            d = (a - b)**2
        distance = np.sqrt(d)
        return distance

    def predict(self, X_test):

    def score(self, X_test, Y_test):
        predictions = self.predict(X_test)
        return (predictions == Y_test).sum() / len(Y_test)


clf = kNN_Classifier(3)
clf.fit(X_train, Y_train)

prediction = clf.predict(X_test)

score = clf.score(X_test, Y_test)
print('Score: ', score*100, '%')

