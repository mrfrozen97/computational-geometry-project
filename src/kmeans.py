import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


def elbow_method(X, max_k=10):
    inertia = []
    k_values = list(range(1, max_k + 1))

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42)
        model.train(X)
        inertia.append(model.inertia_)

    knee = KneeLocator(k_values, inertia, curve="convex", direction="decreasing")
    optimal_k = knee.elbow

    plt.plot(k_values, inertia, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()

    return optimal_k


class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.converged = False

    def initialize_centroids(self, X):
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def assign_clusters(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            new_centroids[i] = X[labels == i].mean(axis=0)
        return new_centroids

    def train_and_animate(self, X, frame_delay=100, save_path="animation.mp4"):
        self.centroids = self.initialize_centroids(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 10)
        self.scatter_points = ax.scatter(X[:, 0], X[:, 1], c=self.labels_, s=50, cmap='viridis')
        self.scatter_center = ax.scatter(self.centroids[:, 0], self.centroids[:, 1],
                                         c='red', s=200, alpha=0.75, marker='X')

        ani_holder = {}

        def update(frame):
            if self.converged:
                return self.scatter_points, self.scatter_center
            self.labels_ = self.assign_clusters(X)
            new_centroids = self.update_centroids(X, self.labels_)

            # Stop if converged
            if np.allclose(self.centroids, new_centroids):
                self.converged = True
                if 'ani' in ani_holder:
                    ani_holder['ani'].event_source.stop()

            self.scatter_points.remove()
            self.scatter_center.remove()
            self.scatter_points = ax.scatter(X[:, 0], X[:, 1], c=self.labels_, s=50, cmap='viridis')
            self.scatter_center = ax.scatter(self.centroids[:, 0], self.centroids[:, 1],
                                             c='red', s=200, alpha=0.75, marker='X')
            self.centroids = new_centroids
            return self.scatter_points, self.scatter_center

        plt.title("K-Means Clustering")
        ani_holder['ani'] = FuncAnimation(
            fig,
            update,
            frames=10,
            blit=True,
            interval=frame_delay,
            repeat=False,
        )
        if save_path and ".gif" in save_path:
            ani_holder['ani'].save(save_path, writer=PillowWriter(fps=10))
        plt.show()
        plt.close(fig)

        self.inertia_ = 0.0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels_ == i]
            self.inertia_ += np.sum((cluster_points - self.centroids[i]) ** 2)

    def train(self, X):
        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iter):
            self.labels_ = self.assign_clusters(X)
            new_centroids = self.update_centroids(X, self.labels_)
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        self.inertia_ = 0.0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels_ == i]
            self.inertia_ += np.sum((cluster_points - self.centroids[i]) ** 2)

    def predict(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def plot_cluster(self, X):
        plt.scatter(X[:, 0], X[:, 1], c=self.labels_, s=50, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1],
                    c='red', s=200, alpha=0.75, marker='X')
        plt.title("K-Means Clustering")
        plt.show()
