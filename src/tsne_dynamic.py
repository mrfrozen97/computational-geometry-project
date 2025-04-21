import numpy as np
from matplotlib import pyplot as plt

from src.base_tsne import BaseTSNE, compute_q_similarities, compute_gradients
from src.tsne_custom_metrics import CustomMetrics


class TSNEDynamic(BaseTSNE):
    def __init__(self, data, n_components=2, perplexity=30.0,
                 learning_rate=200, target_score=0.95,
                 check_interval=50, patience=5, max_iter=5000):
        super().__init__(data, n_components, perplexity, learning_rate, max_iter)
        self.target_score = target_score
        self.check_interval = check_interval
        self.patience = patience
        self.best_Y = None
        self.best_score = -np.inf
        self.no_improvement_count = 0
        self.iterations = 0

    def _check_convergence(self, labels):
        current_score = CustomMetrics(self.Y, labels).calculate_cluster_score()

        if current_score > self.best_score:
            self.best_score = current_score
            self.best_Y = self.Y.copy()
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.best_score >= self.target_score:
            print(f"Target score reached: {self.best_score:.4f}")
            return True

        if self.no_improvement_count >= self.patience:
            print(f"Patience exhausted. Best score: {self.best_score:.4f}")
            return True

        return False

    def fit_transform(self, X, labels):
        self._prepare_optimization(X)
        iteration = 0
        prev_loss = 0

        while True:
            Q = compute_q_similarities(self.Y)
            gradients = compute_gradients(self.P, Q, self.Y)

            # Update embedding
            Y_update = self.learning_rate * gradients + self.momentum * (self.Y - self.Y_prev)
            self.Y_prev = self.Y.copy()
            self.Y -= Y_update

            # Convergence checks
            if iteration % self.check_interval == 0:
                if self._check_convergence(labels):
                    break

                # Loss-based fallback check
                loss = np.sum(self.P * np.log((self.P + 1e-12) / (Q + 1e-12)))
                if iteration > 0 and abs(loss - prev_loss) < 1e-6:
                    print(f"Loss stabilized: {loss:.4f}")
                    break
                prev_loss = loss

            # Safety stop
            if iteration >= self.n_iter:
                print(f"Max iterations reached: {self.n_iter}")
                break

            iteration += 1

        self.iterations = iteration

        return self.best_Y if self.best_score > 0 else self.Y

    def visualize_convergence(self, labels):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.best_Y[:, 0], self.best_Y[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.title(f"Final Embedding (Score: {self.best_score:.4f})")
        plt.colorbar()
        plt.show()
