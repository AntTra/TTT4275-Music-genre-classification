import numpy as np
import pandas as pd

# Expectation-Maximization (EM) algorithm for Gaussian Mixture Model (GMM)
class EM:
    def __init__(self, n_components=1, max_iter=100):
        self.n_components = n_components
        self.max_iter = max_iter

    def fit(self, X, Y):
        # Initialize parameters
        self.means = np.random.rand(self.n_components, X.shape[1])
        self.weights = np.ones(self.n_components) / self.n_components
        
        self.X_train = X
        self.Y_train = Y
        self.X_mean = [np.mean(X[Y==i], axis=0) for i in range(10)]
        self.X_centered = []
        
        # Center the data
        for i in range(len(self.X_train)):
            self.X_centered.append(self.X_train[i] - self.X_mean[self.Y_train[i]])
        self.X_centered = np.array(self.X_centered)

        for _ in range(self.max_iter):
            # E-step: calculate responsibilities
            responsibilities = self._e_step(X)

            # M-step: update parameters
            self._m_step(X, responsibilities)
        
        for i in range(10):
            self.covariance_matrix.append(np.cov(self.X_train[Y_train==i], rowvar=False))
        self.covariance_matrix = np.array(self.covariance_matrix)
            
    def _mahalanobis_distance(self, X1, X2): 
        X1genre = self.Y_train[np.where(self.X_train == X1)[0][0]]
        d = np.dot(np.dot(X1-X2,np.linalg.inv(self.covariance_matrix[X1genre])),X1-X2) 
        return np.sqrt(d)
    
    def _e_step(self, X): # Calculate the responsibilities (probabilities) for each component
        n_samples, n_features = X.shape
        reponsibilities = np.zeros((n_samples, self.n_components))
        
        for i in range(self.n_components):
            density = np.zeros(n_samples)
            for j in range(n_samples):
                d = self.mahalanobis_distance(X[j], self.means[i])**2
                det_cov = np.linalg.det(self.covariance_matrix[i])
                # Compute the normalization constant for a multivariate Gaussian in d dimensions.
                norm_const = 1.0 / (((2 * np.pi) ** (n_features / 2)) * np.sqrt(det_cov))
                
                density[j] = self.weights[i] * norm_const * np.exp(-0.5 * d)
                
            reponsibilities[:, i] = density
            
        # Normalize the responsibilities so that the sum over components is 1 for each sample.
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
            
        return responsibilities

    def _m_step(self, X, responsibilities):
        # Update means, covariances, and weights based on responsibilities
        N = responsibilities.sum(axis=0)
        mCov = np.zeros((self.n_components, X.shape[1], X.shape[1]))
        for i in range(self.n_components):
            # Update means
            self.means[i] = np.dot(responsibilities[:, i], X) / N[i]
            
            # Update covariance matrices
            diff = X - self.means[i]
            mCov[i] = np.dot(responsibilities[:, i] * diff.T, diff) / N[i]
            
            # Update weights
            self.weights[i] = N[i] / X.shape[0]
        

class GMM:
    def __init__(self, n_components=1, max_iter=100):
        self.n_components = n_components
        self.max_iter = max_iter
        self.em = EM(n_components, max_iter)

    def fit(self, X):
        self.em.fit(X)

    def predict(self, X):
        # Predict the class labels for the input data
        pass  # Implement prediction logic here

    def score(self, X, y):
        # Calculate the accuracy of the model
        pass  # Implement scoring logic here

# GMM classifier using the EM algorithm
class GMM_Classifier:
    def __init__(self, n_components=1, max_iter=100):
        self.n_components = n_components
        self.max_iter = max_iter
        self.em = EM(n_components, max_iter)

    def fit(self, X):
        self.em.fit(X)

    def predict(self, X):
        # Predict the class labels for the input data
        pass  # Implement prediction logic here

    def score(self, X, y):
        # Calculate the accuracy of the model
        pass  # Implement scoring logic here

# Test out GMM performance from library 
df = pd.read_csv('Classification music/GenreClassData_30s.txt', delimiter='\t')

df.columns = df.columns.str.strip()

selected_features = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo"]
X = df[selected_features].values

# Labels to genres
y = df["GenreID"].values
X_train, X_test = X[:800],X[800:]
Y_train, Y_test = y[:800],y[800:]

GMM = GMM_Classifier()
GMM.fit(X_train, Y_train)
