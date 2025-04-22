import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.datasets import make_blobs
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


class KCenter:
    def __init__(self, n_clusters=3, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def assign_labels(self, X, centroids):
        # Compute distances from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        # Assign the label of the closest centroid
        labels = np.argmin(distances, axis=1)
        return labels

    def initialize_centroids(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        centroids = [X[np.random.randint(n_samples)]]

        for _ in range(self.n_clusters - 1):
            distances = np.array([np.min([np.linalg.norm(x - c) for c in centroids]) for x in X])
            next_centroid = X[np.argmax(distances)]
            centroids.append(next_centroid)
            if _ == self.n_clusters - 2:
                self.animate_cluster_assignments(np.array(centroids), X, Y=self.assign_labels(X, centroids))

        self.centroids = np.array(centroids)

    def initialize_centroids_animation(self, X, save_path="code"):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        centroids = [X[np.random.randint(n_samples)]]

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(centroids)))

        scatter = ax.scatter(X[:, 0], X[:, 1], c=self.assign_labels(X, centroids), s=50, cmap='viridis')
        ax.set_title("K-Center Clustering Animation")

        def init():
            return []
        def update(frames):
            distances = np.array([np.min([np.linalg.norm(x - c) for c in centroids]) for x in X])
            next_centroid = X[np.argmax(distances)]
            centroids.append(next_centroid)
            new_centeroids = np.array(centroids)
            scatter = ax.scatter(X[:, 0], X[:, 1], c=self.assign_labels(X, centroids), s=50, cmap='viridis')
            ax.scatter(new_centeroids[:, 0], new_centeroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
        ani = animation.FuncAnimation(fig, update, frames=(self.n_clusters - 1),
                                      init_func=init, blit=False, interval=500, repeat=False)
        if save_path and ".gif" in save_path:
            ani.save(save_path, writer=PillowWriter(fps=2))
        plt.show()
        self.centroids = np.array(centroids)

    def assign_clusters(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def train(self, X):
        self.initialize_centroids_animation(X)
        self.labels_ = self.assign_clusters(X)

        # Calculate inertia as maximum distance from any point to its centroid
        point_distances = np.sqrt(((X - self.centroids[self.labels_]) ** 2).sum(axis=1))
        self.inertia_ = np.max(point_distances)

    def predict(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def assign_colors(self, Y):
        basic_colors = [
                                'red', 'blue', 'green', 'orange', 'purple',
                                'cyan', 'magenta', 'yellow', 'lime', 'pink',
                                'teal', 'gold', 'brown', 'olive', 'navy',
                                'gray', 'maroon', 'turquoise', 'violet', 'indigo'
                            ]
        Y_colors = []
        for i in Y:
            Y_colors.append(basic_colors[i%20])
        return Y_colors
    def animate_cluster_assignments(self, centroids, X, Y):
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(centroids)))
        Y_colors = self.assign_colors(Y)

        # Plot points and centroids
        scatter = ax.scatter(X[:, 0], X[:, 1], c="black", s=50, cmap='viridis')
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
        ax.set_title("K-Center Clustering Animation")

        lines = []

        def init():
            return []

        def update(frame):
            # Clear previous lines
            for line in lines:
                line.remove()
            lines.clear()

            idx = frame
            centroid = centroids[Y[idx]]
            point = X[idx]

            # Draw gray line
            line = ax.plot([centroid[0], point[0]], [centroid[1], point[1]],
                           color='gray', linestyle='--', linewidth=2)[0]
            lines.append(line)
            ax.scatter(point[0], point[1], c=Y_colors[idx], s=50, cmap='viridis')

            # Change to cluster color after a short pause
            def change_color():
                line.set_color(colors[Y[idx]])
                fig.canvas.draw_idle()

            fig.canvas.new_timer(interval=300, callbacks=[(change_color, [], {})]).start()

            # Remove the line after another short pause
            def remove_line():
                if line in lines:
                    line.remove()
                    lines.remove(line)
                    fig.canvas.draw_idle()

            fig.canvas.new_timer(interval=600, callbacks=[(remove_line, [], {})]).start()

            return lines

        ani = animation.FuncAnimation(fig, update, frames=len(X),
                                      init_func=init, blit=False, interval=10, repeat=False)
        plt.show()



    def plot_cluster(self, X):
        plt.scatter(X[:, 0], X[:, 1], c=self.labels_, s=50, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1],
                    c='red', s=200, alpha=0.75, marker='X')
        plt.title("K-Center Clustering")
        plt.show()

    def elbow_method(self, X, max_k=10):
        max_distances = []
        k_values = range(1, max_k + 1)

        for k in k_values:
            model = KCenter(n_clusters=k, random_state=42)
            model.train(X)
            max_distances.append(model.inertia_)  # Store maximum distance

        # Detect elbow using Kneedle (convex curve)
        knee = KneeLocator(k_values, max_distances,
                           curve="convex", direction="decreasing")
        optimal_k = knee.elbow

        # Plot
        plt.plot(k_values, max_distances, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Maximum Distance')
        plt.title('Elbow Method for K-Center')
        plt.grid(True)
        plt.show()

        return optimal_k


# Example usage
if __name__ == "__main__":
    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.90, random_state=0)

    # Create and fit K-Center model
    kcenter = KCenter(n_clusters=4, random_state=21)
    kcenter.train(X)
    kcenter.plot_cluster(X)
    print("Inertia (Maximum Distance):", kcenter.inertia_)
