#imports
import numpy as np
import pandas as pd
from collections import Counter

# python class with methods for a kNN classifier
class kNN_Classifier:
    def __init__(self, k):
        self.k = k
        self.covariance_matrix= []

    def fit(self, X, Y):
        """
        Fit the model using the training data.
        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        Y : array-like, shape (n_samples,)
            Target labels.
        """
        self.X_train = X
        self.Y_train = Y

        for i in range(10):
            self.covariance_matrix.append(np.cov(self.X_train[Y_train==i], rowvar=False))
        self.covariance_matrix = np.array(self.covariance_matrix)
    
    def _euclidean_distance(self, X1, X2):
        """
        Compute the Euclidean distance between two points.
        Parameters:
            X1 : array-like, shape (n_features,)
            First point.
            X2 : array-like, shape (n_features,)
            Second point.
        Returns:
            distance : float
                The Euclidean distance between X1 and X2.
        """
        d = np.dot(X1-X2,X1-X2)
        return np.sqrt(d)

    def _mahalanobis_distance(self, X1, X2): 
        """
        Compute the Mahalanobis distance between two points.
        Parameters:
            X1 : array-like, shape (n_features,)
            First point.
            X2 : array-like, shape (n_features,)
            Second point.
        Returns:
            distance : float
                The Mahalanobis distance between X1 and X2.
        """
        X1genre = self.Y_train[np.where(self.X_train == X1)[0][0]]
        d = np.dot(np.dot(X1-X2,np.linalg.inv(self.covariance_matrix[X1genre])),X1-X2) 
        return np.sqrt(d)


    def predict(self, X_test,metric='euclidean'):
        """
        Predict the class labels for the provided test data.
        Parameters:
            X_test : array-like, shape (n_samples, n_features)
                Test data.
            metric : str, optional (default='euclidean')
                The distance metric to use ('euclidean' or 'mahalanobis').
        Returns:
            predictions : array-like, shape (n_samples,)
                Predicted class labels for the test data.
        """
        # Selects distance function
        if metric == 'euclidean':
            distance_func = self._euclidean_distance
        elif metric == 'mahalanobis':
            distance_func = self._mahalanobis_distance
        else:
            return 0
        final_output = []
        for i in range(len(X_test)):
            distances = []
            for j in range(len(self.X_train)):
                distance = distance_func(self.X_train[j] , X_test[i])
                distances.append((distance, j))
            # Sort distances and select KNN
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            # Get labels of neighbors and majority voting
            votes = [self.Y_train[j] for (_, j) in k_nearest]
            ans = Counter(votes).most_common(1)[0][0]
            final_output.append(ans)
            
        return final_output

    def score(self, predictions, Y_test):
        """
        Compute the accuracy of the predictions.
        Parameters:
            predictions : array-like, shape (n_samples,)
                Predicted class labels.
            Y_test : array-like, shape (n_samples,)
                True class labels.
        Returns:
            accuracy : float
                The accuracy of the predictions.
        """
        Y_test = np.array(Y_test)
        return (predictions == Y_test).mean()

    def confusion_matrix(self, predictions, Y_test):
        """
        Compute the confusion matrix.
        Parameters:
            predictions : array-like, shape (n_samples,)
                Predicted class labels.
            Y_test : array-like, shape (n_samples,)
                True class labels.
        Returns:
            cm : array-like, shape (n_classes, n_classes)
                The confusion matrix.
        """
        Y_test = np.array(Y_test)
        cm = np.zeros((10, 10), dtype=int)
        for i in range(len(predictions)):
            cm[Y_test[i]][predictions[i]] += 1
        return cm
    
# reads data from file
df = pd.read_csv('Classification music/GenreClassData_30s.txt', delimiter='\t')
# Whitespace removal
df.columns = df.columns.str.strip()

selected_features = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "mfcc_4_std"]

#exracting data from the dataframe
train_df = df[df["Type"] == "Train"]
test_df = df[df["Type"] == "Test"]
X_train = train_df[selected_features].values
Y_train = train_df["GenreID"].values  
X_test  = test_df[selected_features].values
Y_test  = test_df["GenreID"].values

# fits data to the model
knn = kNN_Classifier(k=8)
knn.fit(X_train, Y_train)

#predicts the data using euclidean and mahalanobis distance
euclidean_prediction = knn.predict(X_test, metric='euclidean')
mahalanobis_prediction = knn.predict(X_test, metric='mahalanobis')
euclidean_confusion_matrix = knn.confusion_matrix(euclidean_prediction, Y_test)
mahalanobis_confusion_matrix = knn.confusion_matrix(mahalanobis_prediction, Y_test)
euclidean_score = knn.score(euclidean_prediction, Y_test)
mahalanobis_score = knn.score(mahalanobis_prediction, Y_test)

#prints the results
print('selected features: ', selected_features)
print('Accuracy for ten genres (Euclidean): ', euclidean_score*100, '%')
print('Accuracy for ten genres (Mahalanobis): ', mahalanobis_score*100, '%')
print('Confusion Matrix (Euclidean): \n',euclidean_confusion_matrix)
print('Confusion Matrix (Mahalanobis): \n',mahalanobis_confusion_matrix)