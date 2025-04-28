import numpy as np
import pandas as pd

# Expectation-Maximization (EM) algorithm for Gaussian Mixture Model (GMM)
class EM:
    def __init__(self, n_components=10, max_iter=100): 
        """
        Initialize the EM algorithm for GMM.

        Parameters:
            n_components : int
                Number of mixture components (e.g., genres).
            max_iter : int
                Maximum number of EM iterations.
        """
        self.n_components = n_components
        self.max_iter = max_iter

    def fit(self, X, Y): # Training data, genre labels(integer)
        """
        Fit the model using the training data.

        Parameters:
            X : array-like, shape (n_samples, n_features)
                Training data.
            Y : array-like, shape (n_samples,)
                Target labels (component indices).
        """
        self.X_train = X
        self.Y_train = Y
        n_samples, n_features = X.shape
        
        self.X_mean = []
        self.num_classes = self.n_components
        self.covariance_matrix = []
        
        for i in range(self.num_classes):
            self.X_mean.append(np.mean(self.X_train[self.Y_train == i], axis=0))

        for i in range(self.num_classes):
            self.covariance_matrix.append(np.cov(self.X_train[self.Y_train == i], rowvar=False))
        self.covariance_matrix = np.array(self.covariance_matrix)
            
        self.means = self.X_mean.copy()  # shape: (n_components, n_features)
        self.weights = np.ones(self.n_components) / self.n_components
 
        for iteration in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            
    def _mahalanobis_distance(self, X1, X2, genre): 
        """
        Compute the Mahalanobis distance between two points.

        Parameters:
            X1 : array-like, shape (n_features,)
            X2 : array-like, shape (n_features,)
            genre : int, component index to select covariance

        Returns:
            float : Mahalanobis distance
        """
        # X1genre = self.Y_train[np.where(self.X_train == X1)[0][0]]
        d = np.dot(np.dot(X1-X2,np.linalg.inv(self.covariance_matrix[genre])),X1-X2) 
        return np.sqrt(d)
    
    def _e_step(self, X): # Calculate the responsibilities (probabilities) for each component
        n_samples, n_features = X.shape
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for i in range(self.n_components):
            density = np.zeros(n_samples)
            for j in range(n_samples):
                genre = self.Y_train[j]
                
                d = self._mahalanobis_distance(self.X_train[j], self.means[i], genre)**2
                det_cov = np.linalg.det(self.covariance_matrix[genre])
                norm_const = 1.0 / (((2 * np.pi) ** (n_features / 2)) * np.sqrt(det_cov))
                
                density[j] = self.weights[i] * norm_const * np.exp(-0.5 * d)
                
            responsibilities[:, i] = density
            
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
            
        return responsibilities

    def _m_step(self, X, responsibilities): # Update means, covariances, and weights based on responsibilities
        N = responsibilities.sum(axis=0)
        n_samples, n_features = X.shape
        mCov = np.zeros((self.n_components, X.shape[1], X.shape[1]))
        
        for i in range(self.n_components):
            self.means[i] = np.dot(responsibilities[:, i], X) / N[i]
            
            #diff = X - self.means[i]
            #mCov[i] = np.dot(responsibilities[:, i] * diff.T, diff) / N[i] # Hahaha lol saken vår er heldigvis supervised 
            
            self.weights[i] = N[i] / n_samples

# GMM classifier using the EM algorithm
class GMM_Classifier:
    def __init__(self, n_components=10, max_iter=100):
        self.n_components = n_components
        self.max_iter = max_iter
        self.em = EM(n_components, max_iter)

    def fit(self, X, Y):
        self.em.fit(X, Y)
        self.num_classes = self.em.n_components

    def predict(self, X_test):
        # Predict the class labels for the input data
        n_samples, n_features = X_test.shape
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for i in range(self.n_components):
            density = np.zeros(n_samples)
            cov = self.em.covariance_matrix[i]
            mean = self.em.means[i]
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
            norm_const = 1.0 / (((2 * np.pi) ** (n_features / 2)) * np.sqrt(det_cov))
            for j in range(n_samples):
                diff = X_test[j] - mean
                d2 = np.dot(np.dot(diff, inv_cov), diff)
                density[j] = self.em.weights[i] * norm_const * np.exp(-0.5 * d2)
            responsibilities[:, i] = density
            
        # Posterior probabilities
        responsibilities /= (responsibilities.sum(axis=1, keepdims=True))
        
        # Predict the genre  with the highest probability.
        predicted_labels = np.argmax(responsibilities, axis=1)
        return predicted_labels

    def score(self, X, y):
        # Calculate the accuracy of the model
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def confusion_matrix(self, predictions, Y_test):
        Y_test = np.array(Y_test)
        cm = np.zeros((10, 10), dtype=int)
        for i in range(len(predictions)):
            cm[Y_test[i]][predictions[i]] += 1
        return cm

df = pd.read_csv('Classification music/GenreClassData_30s.txt', delimiter='\t')
df.columns = df.columns.str.strip()
#0=pop, 1=metal, 2=disco, 3=blues, 4=reggae, 5=classical, 6=rock, 7=hihop, 8=country and 9=jazz
selected_features = ["spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var", "spectral_rolloff_mean","spectral_rolloff_var","spectral_flatness_mean","spectral_contrast_mean",
                     "mfcc_1_mean", "mfcc_2_mean", "mfcc_3_mean", "mfcc_4_mean", "mfcc_5_mean","mfcc_6_mean", "mfcc_7_mean","mfcc_8_mean", "mfcc_9_mean", "mfcc_10_mean", "mfcc_11_mean", "mfcc_12_mean",
                     "chroma_stft_1_mean", "chroma_stft_2_mean", "chroma_stft_3_mean", "chroma_stft_4_mean", "chroma_stft_5_mean", "chroma_stft_6_mean", "chroma_stft_7_mean", "chroma_stft_8_mean", "chroma_stft_9_mean", "chroma_stft_10_mean", "chroma_stft_11_mean", "chroma_stft_12_mean",
                     "mfcc_1_std", "mfcc_2_std", "mfcc_3_std", "mfcc_4_std", "mfcc_5_std", "mfcc_6_std", "mfcc_7_std", "mfcc_8_std", "mfcc_9_std", "mfcc_10_std", "mfcc_11_std", "mfcc_12_std",
                     "tempo"]

train_df = df[df["Type"] == "Train"]
test_df = df[df["Type"] == "Test"]

X_train = train_df[selected_features].values
Y_train = train_df["GenreID"].values  
X_test  = test_df[selected_features].values
Y_test  = test_df["GenreID"].values

gmm = GMM_Classifier(n_components=10, max_iter=100)
gmm.fit(X_train, Y_train)

y_pred = gmm.predict(X_test)
score = gmm.score(X_test, Y_test)
prediction = gmm.predict(X_test)
confusion_matrix = gmm.confusion_matrix(prediction, Y_test)
print("Predictions: ", prediction)
print('Accuracy for ten genres: ', score*100, '%')
print('Confusion Matrix: \n',confusion_matrix)