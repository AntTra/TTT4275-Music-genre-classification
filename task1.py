import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class kNN_Classifier:
    def __init__(self, k):
        self.k = k
        self.covariance_matrix= []

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        self.X_mean = [np.mean(X[Y==i], axis=0) for i in range(10)]
        self.X_centered = []
        # Center the data
        for i in range(len(self.X_train)):
            self.X_centered.append(self.X_train[i] - self.X_mean[self.Y_train[i]])
        self.X_centered = np.array(self.X_centered)

        for i in range(10):
            self.covariance_matrix.append(np.cov(self.X_train[Y_train==i], rowvar=False))
        self.covariance_matrix = np.array(self.covariance_matrix)
    
    def euclidean_distance(self, X1, X2):
        d = np.dot(X1-X2,X1-X2)
        return np.sqrt(d)

    def mahalanobis_distance(self, X1, X2): 
        X1genre = self.Y_train[np.where(self.X_train == X1)[0][0]]
        d = np.dot(np.dot(X1-X2,np.linalg.inv(self.covariance_matrix[X1genre])),X1-X2) 
        return np.sqrt(d)
    # Computational power increases with features and number of potential neighbors
    # Increasing k increases computational power?
    # Large k causes less distinction
    # Small k can be heavily affected by noise
    # Compare Euclidean to other types of distances
    def predict(self, X_test,metric='euclidean'):
        if metric == 'euclidean':
            distance_func = self.euclidean_distance
        elif metric == 'mahalanobis':
            distance_func = self.mahalanobis_distance
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
        Y_test = np.array(Y_test)
        return (predictions == Y_test).mean()

    def confusion_matrix(self, predictions, Y_test):
        Y_test = np.array(Y_test)
        cm = np.zeros((10, 10), dtype=int)
        for i in range(len(predictions)):
            cm[Y_test[i]][predictions[i]] += 1
        return cm
    
df = pd.read_csv('Classification music/GenreClassData_30s.txt', delimiter='\t')
# Whitespace removal
df.columns = df.columns.str.strip()

selected_features = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo"]
#X:data, Y:class of data

train_df = df[df["Type"] == "Train"]
test_df = df[df["Type"] == "Test"]
X_train = train_df[selected_features].values
Y_train = train_df["GenreID"].values  
X_test  = test_df[selected_features].values
Y_test  = test_df["GenreID"].values

knn = kNN_Classifier(k=5)
knn.fit(X_train, Y_train)

euclidean_prediction = knn.predict(X_test, metric='euclidean')
mahalanobis_prediction = knn.predict(X_test, metric='mahalanobis')
euclidean_confusion_matrix = knn.confusion_matrix(euclidean_prediction, Y_test)
mahalanobis_confusion_matrix = knn.confusion_matrix(mahalanobis_prediction, Y_test)
euclidean_score = knn.score(euclidean_prediction, Y_test)
mahalanobis_score = knn.score(mahalanobis_prediction, Y_test)
print('selected features: ', selected_features)
#print("Predictions (Euclidean): ", euclidean_prediction)
#print("Predictions (Mahalanobis): ", mahalanobis_prediction)
print('Accuracy for ten genres (Euclidean): ', euclidean_score*100, '%')
print('Accuracy for ten genres (Mahalanobis): ', mahalanobis_score*100, '%')
print('Confusion Matrix (Euclidean): \n',euclidean_confusion_matrix)
print('Confusion Matrix (Mahalanobis): \n',mahalanobis_confusion_matrix)



# selected_genres = ["pop", "disco", "metal", "classical"]#, "hiphop", "reggae", "blues", "rock", "jazz", "country"]
# # Filter data by selected genres
# data_filtered = df[df["Genre"].isin(selected_genres)]

# Calculate summary statistics grouped by Genre
# summary_stats = data_filtered.groupby("Genre")[selected_features].describe()
# print("Summary Statistics by Genre:")
# print(summary_stats)
# print(knn.covariance())
# #Plot PDF of genres and features 
# plt.figure(figsize=(14, 12))
# for i, feature in enumerate(selected_features):
#     ax = plt.subplot(2, 2, i+1)
#     for genre in selected_genres:
#         subset = data_filtered[data_filtered["Genre"] == genre]
#         sns.kdeplot(subset[feature], label=genre, fill=True, common_norm=False, ax=ax)
#         ax.set_title("")
#     #plt.title(f"PDF of {feature}")
#     #plt.xlabel(feature)
#     plt.ylabel("Density")
# #plt.tight_layout()
# plt.legend()
# plt.show()


