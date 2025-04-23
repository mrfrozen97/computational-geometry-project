import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from matplotlib.animation import PillowWriter


def assign_labels(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels


def elbow_method(X, max_k=10):
    max_distances = []
    k_values = range(1, max_k + 1)

    for k in k_values:
        model = KCenter(n_clusters=k, random_state=42)
        model.train(X)
        max_distances.append(model.inertia_)

    knee = KneeLocator(k_values, max_distances,
                       curve="convex", direction="decreasing")
    optimal_k = knee.elbow
    plt.plot(k_values, max_distances, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Maximum Distance')
    plt.title('Elbow Method for K-Center')
    plt.grid(True)
    plt.show()

    return optimal_k


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

    def initialize_centroids_animation_cluster_assignments(self, X, save_path):
        def assign_colors(Y):
            basic_colors = [
                'red', 'blue', 'green', 'orange', 'purple',
                'cyan', 'magenta', 'yellow', 'lime', 'pink',
                'teal', 'gold', 'brown', 'olive', 'navy',
                'gray', 'maroon', 'turquoise', 'violet', 'indigo'
            ]
            Y_colors = []
            for i in Y:
                Y_colors.append(basic_colors[i % 20])
            return Y_colors

        def animate_cluster_assignments(centroids, X, Y):
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = plt.cm.viridis(np.linspace(0, 1, len(centroids)))
            Y_colors = assign_colors(Y)

            scatter = ax.scatter(X[:, 0], X[:, 1], c="black", s=50, cmap='viridis')
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
            ax.set_title("K-Center Clustering Animation")

            lines = []

            def init():
                return []

            def update(frame):
                for line in lines:
                    line.remove()
                lines.clear()

                idx = frame
                centroid = centroids[Y[idx]]
                point = X[idx]

                line = ax.plot([centroid[0], point[0]], [centroid[1], point[1]],
                               color='gray', linestyle='--', linewidth=2)[0]
                lines.append(line)
                ax.scatter(point[0], point[1], c=Y_colors[idx], s=50, cmap='viridis')

                def change_color():
                    line.set_color(colors[Y[idx]])
                    fig.canvas.draw_idle()

                fig.canvas.new_timer(interval=300, callbacks=[(change_color, [], {})]).start()

                def remove_line():
                    if line in lines:
                        line.remove()
                        lines.remove(line)
                        fig.canvas.draw_idle()

                fig.canvas.new_timer(interval=600, callbacks=[(remove_line, [], {})]).start()

                return lines

            ani = animation.FuncAnimation(fig, update, frames=len(X),
                                          init_func=init, blit=False, interval=10, repeat=False)

            if save_path and ".gif" in save_path:
                ani.save(save_path, writer=PillowWriter(fps=2))
            plt.show()

        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        centroids = [X[np.random.randint(n_samples)]]

        for _ in range(self.n_clusters - 1):
            distances = np.array([np.min([np.linalg.norm(x - c) for c in centroids]) for x in X])
            next_centroid = X[np.argmax(distances)]
            centroids.append(next_centroid)
            if _ == self.n_clusters - 2:
                animate_cluster_assignments(np.array(centroids), X, Y=assign_labels(X, centroids))

        self.centroids = np.array(centroids)

    def initialize_centroids_animation(self, X, save_path):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        centroids = [X[np.random.randint(n_samples)]]

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(centroids)))

        scatter = ax.scatter(X[:, 0], X[:, 1], c=assign_labels(X, centroids), s=50, cmap='viridis')
        ax.set_title("K-Center Clustering Animation")

        def init():
            return []

        def update(frames):
            distances = np.array([np.min([np.linalg.norm(x - c) for c in centroids]) for x in X])
            next_centroid = X[np.argmax(distances)]
            centroids.append(next_centroid)
            new_centeroids = np.array(centroids)
            scatter = ax.scatter(X[:, 0], X[:, 1], c=assign_labels(X, centroids), s=50, cmap='viridis')
            ax.scatter(new_centeroids[:, 0], new_centeroids[:, 1], c='red', s=200, alpha=0.75, marker='X')

        ani = animation.FuncAnimation(fig, update, frames=(self.n_clusters - 1),
                                      init_func=init, blit=False, interval=500, repeat=False)
        if save_path and ".gif" in save_path:
            ani.save(save_path, writer=PillowWriter(fps=2))
        plt.show()
        self.centroids = np.array(centroids)

    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def train_and_animate(self, X, save_path=None):
        self.initialize_centroids_animation(X, save_path)
        self.labels_ = self.assign_clusters(X)

        point_distances = np.linalg.norm(X - self.centroids[self.labels_], axis=1)
        self.inertia_ = np.max(point_distances)

    def train_and_animate_cluster_assignments(self, X, save_path=None):
        self.initialize_centroids_animation_cluster_assignments(X, save_path)
        self.labels_ = self.assign_clusters(X)

        point_distances = np.linalg.norm(X - self.centroids[self.labels_], axis=1)
        self.inertia_ = np.max(point_distances)

    def train(self, X):
        self.initialize_centroids(X)
        self.labels_ = self.assign_clusters(X)

        point_distances = np.linalg.norm(X - self.centroids[self.labels_], axis=1)
        self.inertia_ = np.max(point_distances)

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def plot_cluster(self, X):
        plt.scatter(X[:, 0], X[:, 1], c=self.labels_, s=50, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1],
                    c='red', s=200, alpha=0.75, marker='X')
        plt.title("K-Center Clustering")
        plt.show()
