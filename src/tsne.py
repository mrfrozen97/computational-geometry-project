import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.base_tsne import BaseTSNE, compute_q_similarities, compute_gradients


class TSNE(BaseTSNE):
    def fit_transform_without_graph(self, X):
        self._prepare_optimization(X)
        prev_loss = 0
        for iteration in range(self.n_iter):
            Q = compute_q_similarities(self.Y)
            gradients = compute_gradients(self.P, Q, self.Y)

            # Update with momentum
            Y_update = self.learning_rate * gradients + self.momentum * (self.Y - self.Y_prev)
            self.Y_prev = self.Y.copy()
            self.Y -= Y_update

            if iteration % 100 == 0 or iteration == self.n_iter - 1:
                loss = np.sum(self.P * np.log((self.P + 1e-12) / (Q + 1e-12)))
                if iteration > 0 and abs(loss - prev_loss) < 1e-6:
                    print(f"Converged at iteration {iteration}")
                    return self.Y
                prev_loss = loss
        return self.Y

    def fit_transform(self, X, class_Y):
        self._prepare_optimization(X)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        scatter = ax.scatter(self.Y[:, 0], self.Y[:, 1], c=class_Y, cmap='viridis')

        def update(frame):
            for _ in range(100):
                Q = compute_q_similarities(self.Y)
                gradients = compute_gradients(self.P, Q, self.Y)
                Y_update = self.learning_rate * gradients + self.momentum * (self.Y - self.Y_prev)
                self.Y_prev = self.Y.copy()
                self.Y -= Y_update
            scatter.set_offsets(self.Y)
            ax.set_xlim(self.Y[:, 0].min() - 1, self.Y[:, 0].max() + 1)
            ax.set_ylim(self.Y[:, 1].min() - 1, self.Y[:, 1].max() + 1)
            return scatter,

        FuncAnimation(fig, update, frames=self.n_iter // 100, interval=50, blit=True)
        plt.show()
        return self.Y
