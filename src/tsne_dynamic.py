import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from base_tsne import BaseTSNE, compute_gradients, compute_q_similarities
from tsne_custom_metrics import CustomMetrics
from sklearn.datasets import make_moons, load_iris, load_wine, fetch_openml, load_breast_cancer, make_swiss_roll


class TSNEDynamic(BaseTSNE):
    def __init__(self, data, n_components=2, perplexity=30.0,
                 learning_rate=200, target_score=0.95,
                 check_interval=50, patience=5, n_iter=5000, cluster_threshold=0.9):
        super().__init__(data, n_components, perplexity, learning_rate, n_iter)
        self.target_score = target_score
        self.check_interval = check_interval
        self.patience = patience
        self.best_Y = None
        self.best_score = -np.inf
        self.no_improvement_count = 0
        self.iteration = 0
        self.cluster_threshold = cluster_threshold

    def fit_transform_without_graph(self, X, labels=None):
        self._prepare_optimization(X)
        prev_loss = 0

        for iteration in range(self.n_iter):
            Q = compute_q_similarities(self.Y)
            gradients = compute_gradients(self.P, Q, self.Y)

            # Update with momentum
            Y_update = self.learning_rate * gradients + self.momentum * (self.Y - self.Y_prev)
            self.Y_prev = self.Y.copy()
            self.Y -= Y_update

            # Dynamic convergence checks
            if labels is not None and iteration % self.check_interval == 0:
                metric = CustomMetrics(self.Y, labels)
                current_score = metric.cluster_splitting_without_graph()

                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_Y = self.Y.copy()
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                print(current_score, self.no_improvement_count)

                if self.best_score >= self.cluster_threshold:
                    print(f"Converged by cluster score {self.best_score:.4f} at iteration {iteration}")
                    return self.best_Y

                # Scope for improvement
                # if self.no_improvement_count >= self.patience:
                #     print(f"Stopping early - no improvement for {self.patience} checks")
                #     return self.best_Y

            # Original loss-based convergence
            if iteration % 100 == 0 or iteration == self.n_iter - 1:
                loss = np.sum(self.P * np.log((self.P + 1e-12) / (Q + 1e-12)))
                if iteration > 0 and np.abs(loss - prev_loss) < 1e-6:
                    print("Converged by loss stabilization!")
                    return self.Y
                prev_loss = loss

        return self.best_Y if self.best_score > 0 else self.Y

    def fit_transform(self, X, class_Y):
        self._prepare_optimization(X)
        self.animation_RPS = 100
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        scatter = ax.scatter(self.Y[:, 0], self.Y[:, 1], c=class_Y, cmap='viridis')

        # self.best_Y = self.Y.copy()

        def init():
            scatter.set_offsets(np.empty((0, 2)))  # Empty array
            return scatter,

        def update(iteration):
            for i in range(self.animation_RPS):
                Q = compute_q_similarities(self.Y)
                print("." * int(i * 20 / self.animation_RPS), end="\r")
                gradients = compute_gradients(self.P, Q, self.Y)
                # Update with momentum
                Y_update = self.learning_rate * gradients + self.momentum * (self.Y - self.Y_prev)
                self.Y_prev = self.Y.copy()
                self.Y -= Y_update

                # Dynamic convergence check
                global_iter = iteration * self.animation_RPS + i
                if global_iter % self.check_interval == 0:
                    metric = CustomMetrics(self.Y, class_Y)
                    current_score = metric.cluster_splitting_without_graph()
                    if current_score > self.best_score:
                        self.best_score = current_score
                        self.best_Y = self.Y.copy()
                        self.no_improvement_count = 0
                    else:
                        self.no_improvement_count += 1

                    # Scope for improvement
                    # if self.best_score >= self.cluster_threshold or self.no_improvement_count >= self.patience:
                    #     print(f"\nEarly stop at frame {iteration}")
                    #     scatter.set_offsets(self.best_Y)
                    #     return scatter,

            # Update visualization with best found embedding
            ax.set_xlim(min(self.Y[:, 0]) - 0.1, max(self.Y[:, 0]) + 0.1)
            ax.set_ylim(min(self.Y[:, 1]) - 0.1, max(self.Y[:, 1]) + 0.1)
            scatter.set_offsets(np.column_stack((self.Y[:, 0], self.Y[:, 1])))
            return scatter,

        frames = int(self.n_iter / self.animation_RPS)
        ani = FuncAnimation(
            fig,
            update,
            frames=frames,
            init_func=init,
            blit=True,
            interval=10,
            repeat=False
        )

        plt.title("Random Moving Points")
        plt.show()
        return self.Y



if __name__ == "__main__":
    # Iris dataset
    # Load dataset
    iris = load_iris()
    # iris = load_wine()
    X = iris.data
    y = iris.target

    # # Wine dataset
    # wine = load_wine()
    # X = wine.data
    # y = wine.target
    #
    # # Breast Cancer dataset
    # cancer = load_breast_cancer()
    # X = cancer.data
    # y = cancer.target


    # Run your t-SNE
    tsne = TSNEDynamic(data=X, n_components=2, perplexity=50, learning_rate=200, n_iter=20000)
    X_embedded = tsne.fit_transform_without_graph(X, y)

    # Plot the 2D projection
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis')
    plt.title("t-SNE visualization of Iris dataset")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter)
    plt.grid(True)
    plt.show()
