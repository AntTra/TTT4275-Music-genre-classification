import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
# import tensorflow as tf
# from tensorflow.keras.models import Sequential


def load_and_prepare_data():
    files = ['Classification music/GenreClassData_5s.txt', 
             'Classification music/GenreClassData_10s.txt', 
             'Classification music/GenreClassData_30s.txt']
    
    train_data_list = []
    test_data_list = []

    datasets = []
    for file in files:
        df = pd.read_csv(file, delimiter='\t')
        df.columns = df.columns.str.strip()  # Remove whitespace
        test_df = df[df["Type"] == "test"]
        train_df = df[df["Type"] == "train"]
        train_df = train_df.drop(columns=["File", "Genre", "GenreID", "Type"])  # Drop unnecessary columns
        test_df = test_df.drop(columns=["File", "Genre", "GenreID", "Type"])  # Drop unnecessary columns
        train_labels = train_df["GenreID"].values  # Extract labels
        test_labels = test_df["GenreID"].values  # Extract labels
        train_data_list.append(df.values)
    
    for data_list, labels in [[train_data_list, train_labels], [test_data_list, test_labels]]:
        data_5s, data_10s, data_30s = data_list

        # Repeat data to match sequence length
        data_10s_repeated = np.repeat(data_10s, 2, axis=0)
        data_30s_repeated = np.repeat(data_30s, 6, axis=0)
    #hver snag har 1 entry i 30s, 3 i 10s, 6 i 5s
        print(data_5s.shape, data_10s_repeated.shape, data_30s_repeated.shape)
        
        # Concatenate along sequence length dimension
        combined_data = np.concatenate((data_5s, data_10s_repeated, data_30s_repeated), axis=1)
        print(combined_data.shape)

        # Remove the Track ID column
        combined_data = np.delete(combined_data, 0, axis=1)
        combined_data = np.delete(combined_data, 63, axis=1)
        combined_data = np.delete(combined_data, 126, axis=1)
        print(combined_data[0])
        
        datasets.append(np.concatinate(combined_data, labels, axis=1))
    
    return datasets[0], datasets[1]

# Load and prepare the data
data_ndarray = load_and_prepare_data()
print("Data shape:", data_ndarray.shape)

tfrain_ds, test_ds = tf.data.Dataset.from_tensor_slices(data_ndarray)

train_ds.window(6, shift=1, stride=6, drop_remainder=True).flat_map(
    lambda w: w.batch(6)).map(lambda w: (w[:, :-1], w[:, 1:])
).batch(32).prefetch(tf.data.AUTOTUNE)

train_ds = train_ds.shuffle(buffer_size=100, reshuffle_each_iteration=False)

#Feed forward neural network with softmax activation function
model = Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(6, 6)))

for _ in range(3):
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

for i in range(3):
    model.add(tf.keras.layers.Dense(32 / (2^(i)), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

model.summary()

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.fit(train_ds, epochs=10, verbose=1)
model.evaluate(test_ds, verbose=1)





 
class kNN_Classifier:
    def __init__(self, k):
        self.k = k
        self.covariance_matrix = []

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        self.X_mean = [np.mean(X[Y==i], axis=0) for i in range(10)]
        self.X_centered = []
        # Center the data
        for i in range(len(self.X_train)):
            self.X_centered.append(self.X_train[i] - self.X_mean[self.Y_train[i]])
        self.X_centered = np.array(self.X_centered)

        self.X_mean = [np.mean(X[Y==i], axis=0) for i in range(10)]
        self.X_centered = []
        # Center the data
        for i in range(len(self.X_train)):
            self.X_centered.append(self.X_train[i] - self.X_mean[self.Y_train[i]])
        self.X_centered = np.array(self.X_centered)

        for i in range(10):
            self.covariance_matrix.append(np.cov(self.X_train[Y_train==i], rowvar=False))
        self.covariance_matrix = np.array(self.covariance_matrix)
        self.covariance_matrix = np.array(self.covariance_matrix)

    
    def euclidean_distance(self, X1, X2):
        d = sum((a - b)**2 for a, b in zip(X1, X2))
        return np.sqrt(d)

    def mahalanobis_distance(self, X1, X2): #must make actual distance  
        X1genre = self.Y_train[np.where(self.X_train == X1)[0][0]]
        d = np.dot(np.dot(X1-X2,np.linalg.inv(self.covariance_matrix[X1genre])),X1-X2) #sum(np.dot(a - b, np.linalg.inv(self.covariance_matrix[a])) for a, b in zip(X1, X2))

    def mahalanobis_distance(self, X1, X2): #must make actual distance  
        X1genre = self.Y_train[np.where(self.X_train == X1)[0][0]]
        d = np.dot(np.dot(X1-X2,np.linalg.inv(self.covariance_matrix[X1genre])),X1-X2) #sum(np.dot(a - b, np.linalg.inv(self.covariance_matrix[a])) for a, b in zip(X1, X2))

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

    def score(self, predictions, Y_test):
        Y_test = np.array(Y_test)
        return (predictions == Y_test).mean()
    
    def confusion_matrix(self, predictions, Y_test):
        Y_test = np.array(Y_test)
        cm = np.zeros((10, 10), dtype=int)
        for i in range(len(predictions)):
            cm[Y_test[i]][predictions[i]] += 1
        return cm

class mahalanobis_Classifier:
    def __init__(self):
        self.covariance_matrix = []
    def fit(self, X,Y):
        self.X_train = X
        self.Y_train = Y
        self.X_mean = [np.mean(X[Y==i], axis=0) for i in range(10)]

        for i in range(10):
            self.covariance_matrix.append(np.cov(self.X_train[Y_train==i], rowvar=False))
        self.covariance_matrix = np.array(self.covariance_matrix)

    def mahalanobis_distance(self, X,genre): #must make actual distance  
        d = np.dot(np.dot(X-self.X_mean[genre],np.linalg.inv(self.covariance_matrix[genre])),X-self.X_mean[genre]) #sum(np.dot(a - b, np.linalg.inv(self.covariance_matrix[a])) for a, b in zip(X1, X2))

        return np.sqrt(d)
    
    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            smallest_distance = 1000000
            ans = 0
            for j in range(10):
                distance = self.mahalanobis_distance(X_test[i],j)
                if smallest_distance > distance:
                    smallest_distance = distance
                    ans = j
            predictions.append(ans)
        return predictions
            
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

# TODO: Split code into train and test
selected_features = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "mfcc_4_std"]
X = df[selected_features].values 

# Labels to genres
y = df["GenreID"].values
X_train, X_test = X[:800],X[800:]
Y_train, Y_test = y[:800],y[800:]


knn = kNN_Classifier(k=5)
knn.fit(X_train, Y_train)
prediction = knn.predict(X_test)
confusion_matrix = knn.confusion_matrix(prediction, Y_test)
score = knn.score(prediction, Y_test)
#print("Predictions: ", prediction)
print("================================================")
print("KNN CLASSIFIER")
print('Accuracy for ten genres: ', score*100, '%')
print('Confusion Matrix: \n',confusion_matrix)

mahala = mahalanobis_Classifier()
mahala.fit(X_train,Y_train)
prediction = mahala.predict(X_test)
confusion_matrix = mahala.confusion_matrix(prediction, Y_test)
score = mahala.score(prediction, Y_test)
#print("Predictions: ", prediction)
print("================================================")
print("MAHALANOBIS CLASSIFIER")
print('Accuracy for ten genres: ', score*100, '%')
print('Confusion Matrix: \n',confusion_matrix)


# selected_genres = ["pop", "disco", "metal", "classical"]#, "hiphop", "reggae", "blues", "rock", "jazz", "country"]
# # Filter data by selected genres
# data_filtered = df[df["Genre"].isin(selected_genres)]
print('Confusion Matrix: \n',confusion_matrix)


# selected_genres = ["pop", "disco", "metal", "classical"]#, "hiphop", "reggae", "blues", "rock", "jazz", "country"]
# # Filter data by selected genres
# data_filtered = df[df["Genre"].isin(selected_genres)]

# # Calculate summary statistics grouped by Genre 
# summary_stats = data_filtered.groupby("Genre")[selected_features].describe()
# print("Summary Statistics by Genre:")
# print(summary_stats)
# print(knn.covariance())
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