import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class kNN_Classifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def euclidean_distance(self, X1, X2):
        d = sum((a - b)**2 for a, b in zip(X1, X2))
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
                distance = self.euclidean_distance(self.X_train[j] , X_test[i])
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

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = kNN_Classifier(k=5)
knn.fit(X_train, Y_train)

prediction = knn.predict(X_test)

score = knn.score(X_test, Y_test)
print("Predictions: ", prediction)
print('Accuracy for ten genres: ', score*100, '%')

selected_genres = ["pop", "disco", "metal", "classical"]
data_filtered = df[df["Genre"].isin(selected_genres)]

# Calculate summary statistics grouped by Genre
summary_stats = data_filtered.groupby("Genre")[selected_features].describe()
print("Summary Statistics by Genre:")
print(summary_stats)

# Plot PDF of genres and features 
plt.figure(figsize=(14, 12))
for i, feature in enumerate(selected_features):
    plt.subplot(2, 2, i+1)
    for genre in selected_genres:
        subset = data_filtered[data_filtered["Genre"] == genre]
        sns.kdeplot(subset[feature], label=genre, fill=True, common_norm=False)
    plt.title(f"PDF of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
plt.tight_layout()
plt.show()





