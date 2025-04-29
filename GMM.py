import numpy as np
import pandas as pd

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
        self.max_iter     = max_iter

    def fit(self, X, Y):
        """
        Fit the model using the training data.

        Parameters:
            X : array-like, shape (n_samples, n_features)
                Training data.
            Y : array-like, shape (n_samples,)
                Target labels (component indices).
        """
        self.X_train       = X
        self.Y_train       = Y
        n_samples, n_feat = X.shape
        
        self.X_mean = []
        self.num_classes = self.n_components
        self.covariance_matrix = []
        
        for i in range(self.num_classes):
            self.X_mean.append(np.mean(self.X_train[self.Y_train == i], axis=0))

        for i in range(self.num_classes):
            self.covariance_matrix.append(
                np.cov(self.X_train[self.Y_train == i], rowvar=False)
            )
        self.covariance_matrix = np.array(self.covariance_matrix)
            
        self.means   = self.X_mean.copy()
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
        d = np.dot(np.dot(X1 - X2, np.linalg.inv(self.covariance_matrix[genre])),(X1 - X2))
        return np.sqrt(d)

    def _e_step(self, X):# Calculate the responsibilities (probabilities) for each component
        n_samples, n_feat = X.shape
        K = self.n_components
        resp = np.zeros((n_samples, K))

        for k in range(K):
            mu_k     = self.means[k]
            sig_k     = self.covariance_matrix[k]
            invsig_k  = np.linalg.inv(sig_k)
            detsig_k  = np.linalg.det(sig_k)
            norm_k  = self.weights[k] / (
                (2*np.pi)**(n_feat/2) * np.sqrt(detsig_k)
            )

            for n in range(n_samples):
                diff    = X[n] - mu_k
                d2      = diff.dot(invsig_k).dot(diff)
                resp[n, k] = norm_k * np.exp(-0.5 * d2)

        # Normalize so each row sums to 1
        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def _m_step(self, X, responsibilities): # Update means, covariances, and weights based on responsibilities
        N = responsibilities.sum(axis=0)
        n_samples, n_feat = X.shape
        
        for k in range(self.n_components):
            self.means[k]   = (responsibilities[:, k] @ X) / N[k]
            self.weights[k] = N[k] / n_samples

class GMM_Classifier:
    def __init__(self, n_components=10, max_iter=100):
        self.n_components = n_components
        self.max_iter     = max_iter
        self.em           = EM(n_components, max_iter)

    def fit(self, X, Y):
        # exactly your original EM.fit call
        self.em.fit(X, Y)
        self.num_classes = self.em.n_components

    def predict(self, X_test):
        resp = self.em._e_step(X_test)
        return np.argmax(resp, axis=1)

    def score(self, X, y):
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
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

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
print('Accuracy for ten genres: ', score*100, '%')
print('Confusion Matrix: \n',confusion_matrix)