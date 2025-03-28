import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class kNN_Classifier:
    def __init__(self, k):
        self.k = k
        self.covariance_matrix = []

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def covariance(self):
        for i in range(10):
            self.covariance_matrix.append(np.cov(self.X_train[Y_train==i], rowvar=False))
        #print(self.covariance_matrix)

    
    def euclidean_distance(self, X1, X2):
        d = sum((a - b)**2 for a, b in zip(X1, X2))
        return np.sqrt(d)

    def mahalanobis_distance(self, X1, X2): #must make actual distance
        
        d = sum([np.dot(np.dot(X1-X2,np.linalg.inv(self.covariance_matrix[i])),X1-X2) for i in range(10)]) #sum(np.dot(a - b, np.linalg.inv(self.covariance_matrix[a])) for a, b in zip(X1, X2)) ## AAAAAAAAAAAAAAA
         #d = sum(np.dot(a - b, np.linalg.inv(self.covariance_matrix[a])) for a, b in zip(X1, X2)) ## AAAAAAAAAAAAAAA
        #d = np.dot(X1 - X2, np.linalg.inv(self.covariance_matrix[i for i in range(10)]))
        return np.sqrt(d)
    # Computational power increases with features and number of potential neighbors
    # Increasing k increases computational power?
    # Large k causes less distinction
    # Small k can be heavily affected by noise
    # Compare Euclidean to other types of distances
    def predict(self, X_test):
        final_output = []
        for i in range(len(X_test)):
            distances = []
            for j in range(len(self.X_train)):
                #distance = self.euclidean_distance(self.X_train[j] , X_test[i])
                distance = self.mahalanobis_distance(self.X_train[j] , X_test[i])
                distances.append((distance, j))
            # Sort distances and select KNN
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            # Get labels of neighbors and majority voting
            votes = [self.Y_train[j] for (_, j) in k_nearest]
            ans = Counter(votes).most_common(1)[0][0]
            final_output.append(ans)
            
        return final_output

    def score(self, X_test, Y_test):
        predictions = np.array(self.predict(X_test))
        Y_test = np.array(Y_test)
        return (predictions == Y_test).mean()

df = pd.read_csv('Classification music/GenreClassData_30s.txt', delimiter='\t')

# Whitespace removal
df.columns = df.columns.str.strip()

# TODO: Split code into train and test
selected_features = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo"]
X = df[selected_features].values 

# Labels to genres
y = df["GenreID"].values
X_train, X_test = X[:800],X[800:]
Y_train, Y_test = y[:800],y[800:]

knn = kNN_Classifier(k=5)
knn.fit(X_train, Y_train)
knn.covariance()
prediction = knn.predict(X_test)

score = knn.score(X_test, Y_test)
print("Predictions: ", prediction)
print('Accuracy for ten genres: ', score*100, '%')

selected_genres = ["pop", "disco", "metal", "classical"]#, "hiphop", "reggae", "blues", "rock", "jazz", "country"]
# Filter data by selected genres
data_filtered = df[df["Genre"].isin(selected_genres)]

# Calculate summary statistics grouped by Genre
summary_stats = data_filtered.groupby("Genre")[selected_features].describe()
print("Summary Statistics by Genre:")
print(summary_stats)
print(knn.covariance())
#Plot PDF of genres and features 
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
