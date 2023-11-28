import numpy as np
from scipy.spatial.distance import cdist

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        # Number of points
        n_points = X.shape[0]
        # Labels for each point
        labels = np.full(n_points, -1)
        # Visited points
        visited = np.zeros(n_points, dtype=bool)

        # Helper function to find neighbors
        def find_neighbors(i):
            distances = cdist([X[i]], X)[0]
            neighbors = np.where(distances < self.eps)[0]
            return neighbors

        # Iterate over points
        cluster_id = 0
        for i in range(n_points):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = find_neighbors(i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # Noise
            else:
                labels[i] = cluster_id
                k = 0
                while k < len(neighbors):
                    j = neighbors[k]
                    if not visited[j]:
                        visited[j] = True
                        new_neighbors = find_neighbors(j)
                        if len(new_neighbors) >= self.min_samples:
                            neighbors = np.append(neighbors, new_neighbors)
                    if labels[j] == -1:
                        labels[j] = cluster_id
                    k += 1
                cluster_id += 1
        self.labels_ = labels
        return self
