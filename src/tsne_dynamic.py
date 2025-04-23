import numpy as np

from src.base_tsne import BaseTSNE, compute_gradients, compute_q_similarities
from src.tsne_custom_metrics import CustomMetrics


class TSNEDynamic(BaseTSNE):
    def __init__(self, data, n_components=2, perplexity=30.0,
                 learning_rate=200, target_score=0.9,
                 check_interval=50, n_iter=5000, tolerance_percentage=2):
        super().__init__(data, n_components, perplexity, learning_rate, n_iter)
        self.target_score = target_score
        self.check_interval = check_interval
        self.score = -np.inf
        self.tolerance_percentage = tolerance_percentage

    def fit_transform(self, X, labels=None):
        self._prepare_optimization(X)
        prev_loss = 0

        for iteration in range(self.n_iter):
            Q = compute_q_similarities(self.Y)
            gradients = compute_gradients(self.P, Q, self.Y)

            Y_update = self.learning_rate * gradients + self.momentum * (self.Y - self.Y_prev)
            self.Y_prev = self.Y.copy()
            self.Y -= Y_update

            if labels is not None and iteration % self.check_interval == 0:
                metric = CustomMetrics(self.Y, labels, self.tolerance_percentage)
                self.score = metric.cluster_splitting_without_graph()
                if self.score >= self.target_score:
                    print(f"Converged by cluster score {self.score:.4f} at iteration {iteration}")
                    self.iterations = iteration
                    return self.Y

            if iteration % 100 == 0 or iteration == self.n_iter - 1:
                loss = np.sum(self.P * np.log((self.P + 1e-12) / (Q + 1e-12)))
                if iteration > 0 and np.abs(loss - prev_loss) < 1e-6:
                    print("Converged by loss stabilization!")
                    self.iterations = iteration
                    return self.Y
                prev_loss = loss
            self.iterations = iteration

        return self.Y
