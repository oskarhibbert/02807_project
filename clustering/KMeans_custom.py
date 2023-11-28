import numpy as np

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

# KMeans algorithm
class KMeans_custom:
    def __init__(self, K=3, max_iters=100, random_state=42):
        self.K = K
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None

    def initialize_centroids(self, X):
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.K]]
        return centroids

    def assign_clusters(self, X):
        distances = np.array([euclidean_distance(X, centroid) for centroid in self.centroids])
        return np.argmin(distances, axis=0)

    def update_centroids(self, X, clusters):
        centroids = np.array([X[clusters == k].mean(axis=0) for k in range(self.K)])
        return centroids

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iters):
            clusters = self.assign_clusters(X)
            previous_centroids = self.centroids
            self.centroids = self.update_centroids(X, clusters)
            
            # Check for convergence (if centroids do not change)
            if np.all(previous_centroids == self.centroids):
                break

        return clusters