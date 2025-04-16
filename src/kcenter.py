import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


class KCenter:
    def __init__(self, n_clusters=3, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def initialize_centroids(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        centroids = [X[np.random.randint(n_samples)]]

        for _ in range(self.n_clusters - 1):
            distances = np.array([np.min([np.linalg.norm(x - c) for c in centroids]) for x in X])
            next_centroid = X[np.argmax(distances)]
            centroids.append(next_centroid)

        self.centroids = np.array(centroids)

    def assign_clusters(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def train(self, X):
        self.initialize_centroids(X)
        self.labels_ = self.assign_clusters(X)

        # Calculate inertia as maximum distance from any point to its centroid
        point_distances = np.sqrt(((X - self.centroids[self.labels_]) ** 2).sum(axis=1))
        self.inertia_ = np.max(point_distances)

    def predict(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def plot_cluster(self, X):
        plt.scatter(X[:, 0], X[:, 1], c=self.labels_, s=50, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1],
                    c='red', s=200, alpha=0.75, marker='X')
        plt.title("K-Center Clustering")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.90, random_state=0)

    # Create and fit K-Center model
    kcenter = KCenter(n_clusters=4, random_state=21)
    kcenter.train(X)
    kcenter.plot_cluster(X)
    print("Inertia (Maximum Distance):", kcenter.inertia_)
