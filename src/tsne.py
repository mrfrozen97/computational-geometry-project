import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



class Tsne():
    def __init__(self, data, n_components=2, perplexity=30.0, learning_rate=200, n_iter=1000):
        self.X = data
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = 4.0  # Important for initial exploration
        self.momentum = 0.5  # Helps with convergence

    def compute_pairwise_distance(self, X):
        sum_X = np.sum(np.square(X), axis=1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return np.maximum(D, 0)

    def _compute_p_matrix(self, distances):
        def binary_search_perplexity(dist_row, target_perplexity):
            beta_min = -np.inf
            beta_max = np.inf
            beta = 1.0

            for _ in range(50):
                affinities = np.exp(-dist_row * beta)
                sum_affinities = np.sum(affinities)
                if sum_affinities == 0:
                    sum_affinities = 1e-12
                probabilities = affinities / sum_affinities

                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
                perplexity = 2 ** entropy

                perplexity_diff = perplexity - target_perplexity

                if np.abs(perplexity_diff) < 1e-5:
                    break

                if perplexity_diff > 0:
                    beta_min = beta
                    beta = (beta_max + beta) / 2 if beta_max != np.inf else beta * 2
                else:
                    beta_max = beta
                    beta = (beta_min + beta) / 2 if beta_min != -np.inf else beta / 2

            return probabilities

        n = distances.shape[0]
        P = np.zeros((n, n))

        for i in range(n):
            P[i] = binary_search_perplexity(distances[i], self.perplexity)

        P = (P + P.T) / (2 * n)
        P = np.maximum(P, 1e-12)
        P = P / np.sum(P)
        return P

    def compute_q_similarities(self, Y):
        distances = self.compute_pairwise_distance(Y)
        inv_distances = 1.0 / (1.0 + distances)
        np.fill_diagonal(inv_distances, 0)
        Q = inv_distances / np.sum(inv_distances)
        Q = np.maximum(Q, 1e-12)
        return Q

    def _compute_gradients(self, P, Q, Y):
        n, d = Y.shape
        grad = np.zeros((n, d))

        PQ_diff = P - Q
        inv_dist = 1.0 / (1.0 + self.compute_pairwise_distance(Y))

        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4 * np.sum((PQ_diff[:, i] * inv_dist[:, i])[:, np.newaxis] * diff, axis=0)

        return grad

    def fit_transform(self, X, class_Y):
        np.random.seed(42)
        print("hello")
        self.Y = np.random.randn(X.shape[0], self.n_components) * 1e-4
        self.Y_prev = np.zeros_like(self.Y)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        distances = self.compute_pairwise_distance(X)
        self.P = self._compute_p_matrix(distances)
        self.animation_RPS = 100

        # Early exaggeration
        # self.P *= self.early_exaggeration
        #print(self.Y)
        # Set up the figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        scatter = ax.scatter(self.Y[:, 0], self.Y[:, 1], c=class_Y, cmap='viridis')

        def init():
            scatter.set_offsets(np.empty((0, 2)))  # Empty array
            return scatter,


        def update(iteration):
                for i in range(self.animation_RPS):

                    Q = self.compute_q_similarities(self.Y)
                    print("."*int(i*20/self.animation_RPS), end="\r")
                    # Stop early exaggeration after 100 iterations
                    # if iteration == 100:
                    #     self.P /= self.early_exaggeration

                    gradients = self._compute_gradients(self.P, Q, self.Y)

                    # Update with momentum
                    Y_update = self.learning_rate * gradients + self.momentum * (self.Y - self.Y_prev)
                    self.Y_prev = self.Y.copy()
                    self.Y -= Y_update
                #if iteration % 100 == 0 or iteration == self.n_iter - 1:
                # Check for convergence
                    # if iteration > 0 and np.abs(loss - prev_loss) < 1e-6:
                    #     print("Converged!")
                    #     break
                    #prev_loss = loss
                    #print(self.Y)
                ax.set_xlim(min(self.Y[:, 0]) - 0.1, max(self.Y[:, 0]) + 0.1)  # 10% buffer on both sides
                ax.set_ylim(min(self.Y[:, 1]) - 0.1, max(self.Y[:, 1]) + 0.1) # 10% buffer on both sides
                loss = np.sum(self.P * np.log((self.P + 1e-12) / (Q + 1e-12)))
                print(f"Iteration {iteration}: KL Divergence = {loss:.4f}")
                scatter.set_offsets(np.column_stack((self.Y[:, 0], self.Y[:, 1])))
                return scatter,

        frames = int(self.n_iter/self.animation_RPS)
        print(frames)
        # Create animation
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
