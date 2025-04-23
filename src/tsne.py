import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from base_tsne import BaseTSNE, compute_q_similarities, compute_gradients


class TSNE(BaseTSNE):
    def fit_transform(self, X):
        self._prepare_optimization(X)
        prev_loss = 0
        for iteration in range(self.n_iter):
            Q = compute_q_similarities(self.Y)
            gradients = compute_gradients(self.P, Q, self.Y)
            Y_update = self.learning_rate * gradients + self.momentum * (self.Y - self.Y_prev)
            self.Y_prev = self.Y.copy()
            self.Y -= Y_update
            if iteration % 100 == 0 or iteration == self.n_iter - 1:
                print(f"Iteration   {iteration}")
                loss = np.sum(self.P * np.log((self.P + 1e-12) / (Q + 1e-12)))
                if iteration > 0 and np.abs(loss - prev_loss) < 1e-6:
                    print("Converged!")
                    return self.Y
                prev_loss = loss
        return self.Y

    def fit_transform_animated(self, X, class_Y):
        self._prepare_optimization(X)
        self.animation_RPS = 100
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        scatter = ax.scatter(self.Y[:, 0], self.Y[:, 1], c=class_Y, cmap='viridis')

        def init():
            scatter.set_offsets(np.empty((0, 2)))  # Empty array
            return scatter,

        def update(iteration):
            print(iteration)
            for i in range(self.animation_RPS):
                Q = compute_q_similarities(self.Y)
                print("." * int(i * 20 / self.animation_RPS), end="\r")
                gradients = compute_gradients(self.P, Q, self.Y)

                # Update with momentum
                Y_update = self.learning_rate * gradients + self.momentum * (self.Y - self.Y_prev)
                self.Y_prev = self.Y.copy()
                self.Y -= Y_update

            ax.set_xlim(min(self.Y[:, 0]) - 0.1, max(self.Y[:, 0]) + 0.1)
            ax.set_ylim(min(self.Y[:, 1]) - 0.1, max(self.Y[:, 1]) + 0.1)
            loss = np.sum(self.P * np.log((self.P + 1e-12) / (Q + 1e-12)))
            print(f"Iteration {iteration}: KL Divergence = {loss:.4f}")
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
