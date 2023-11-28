import numpy as np
from scipy.spatial.distance import pdist, squareform

class HierarchicalClustering:
    def __init__(self, method='ward'):
        self.method = method
        self.labels_ = None

    def fit(self, X):
        if self.method == 'ward':
            self.linkage_matrix = self.ward_linkage(X)
        else:
            # Implement other methods if needed
            pass
        return self

    def ward_linkage(self, X):
        num_samples = X.shape[0]
        distances = squareform(pdist(X, metric='euclidean'))
        clusters = [[i] for i in range(num_samples)]
        linkage_matrix = []

        for _ in range(num_samples - 1):
            min_dist = np.inf
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist_ij = self.ward_distance(clusters[i], clusters[j], distances)
                    if dist_ij < min_dist:
                        min_dist = dist_ij
                        min_clusters = (i, j)

            i, j = min_clusters
            new_cluster = clusters[i] + clusters[j]
            linkage_matrix.append([i, j, min_dist, len(new_cluster)])
            clusters[i] = new_cluster
            del clusters[j]

        return np.array(linkage_matrix)

    def ward_distance(self, cluster_i, cluster_j, distances):
        dist = 0
        for i in cluster_i:
            for j in cluster_j:
                dist += distances[i, j]
        dist /= len(cluster_i) * len(cluster_j)
        return dist