import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from kneed import KneeLocator
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


# Class Kmeans implements the kmeans clustering algorithm
class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):

        self.n_clusters = n_clusters        # Number of cluster centers/clusters
        self.max_iter = max_iter            # Maximum permitted iterations if the algorithm does not converge
        self.tol = tol                      # The Error value accepted for convergence
        self.random_state = random_state    # Used to initialize random cluster centers

        # Initialize variables required for later
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.converged = False

    # Returns a np array of randomly generated cluster centers
    def initialize_centroids(self, X):
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    # Computes distance between cluster points and the centers. Assign the cluster with minimum distance
    def assign_clusters(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids)**2).sum(axis=2))
        return np.argmin(distances, axis=1)

    # Compute new centers by taking the mean of points in the newly formed clusters
    def update_centroids(self, X, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            new_centroids[i] = X[labels == i].mean(axis=0)
        return new_centroids

    # Run the main K-means algorithm steps while animating it
    def train_and_animate(self, X, frame_delay=100, save_path="animation.mp4"):
        self.centroids = self.initialize_centroids(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 10)
        self.scatter_points = ax.scatter(X[:, 0], X[:, 1], c=self.labels_, s=50, cmap='viridis')
        self.scatter_center = ax.scatter(self.centroids[:, 0], self.centroids[:, 1],
                    c='red', s=200, alpha=0.75, marker='X')

        # Mutable wrapper to access ani inside update
        ani_holder = {}

        def update(frame):
            if self.converged:
                return self.scatter_points, self.scatter_center
            print(frame)
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
                frames=self.max_iter,
                blit=True,
                interval=frame_delay,
                repeat=False,
            )
        if save_path and ".gif" in save_path:
            ani_holder['ani'].save(save_path, writer=PillowWriter(fps=10))

        else:
            plt.show()
        plt.show()
        plt.close(fig)

        # Run the algorithm when either we reach max iterations or the centers don't change


        # Compute inertia (sum of squared distances to closest center)
        self.inertia_ = 0.0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels_ == i]
            self.inertia_ += np.sum((cluster_points - self.centroids[i]) ** 2)


    # Run the main K-means algorithm steps
    def train(self, X):
        self.centroids = self.initialize_centroids(X)

        # Run the algorithm when either we reach max iterations or the centers don't change
        for _ in range(self.max_iter):
            self.labels_ = self.assign_clusters(X)
            new_centroids = self.update_centroids(X, self.labels_)

            # Stop if centroids don't change
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        # Compute inertia (sum of squared distances to closest center)
        self.inertia_ = 0.0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels_ == i]
            self.inertia_ += np.sum((cluster_points - self.centroids[i]) ** 2)

    # Predicts new points based on the closed cluster center
    def predict(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids)**2).sum(axis=2))
        return np.argmin(distances, axis=1)

    # Plots the cluster points and their centeroids
    def plot_cluster(self, X):
        # Visualize results
        plt.scatter(X[:, 0], X[:, 1], c=self.labels_, s=50, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1],
                    c='red', s=200, alpha=0.75, marker='X')
        plt.title("K-Means Clustering")
        plt.show()

    def elbow_method(self, X, max_k=10):
        inertias = []
        k_values = list(range(1, max_k + 1))

        for k in k_values:
            model = KMeans(n_clusters=k, random_state=42)
            model.train(X)
            model.plot_cluster(X)
            inertias.append(model.inertia_)

        # Detect the elbow using Kneedle
        knee = KneeLocator(k_values, inertias, curve="convex", direction="decreasing")
        optimal_k = knee.elbow

        # Plot the elbow graph
        plt.plot(k_values, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (Sum of Squared Distances)')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)
        plt.show()

        return optimal_k



# Example usage
if __name__ == "__main__":
    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.90, random_state=0)

    # Create and fit model
    kmeans = KMeans(n_clusters=4, random_state=21)
    print(kmeans.train_and_animate(X))
    labels = kmeans.labels_

