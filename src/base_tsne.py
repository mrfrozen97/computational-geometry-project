import numpy as np


def compute_pairwise_distance(X):
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return np.maximum(D, 0)


def compute_q_similarities(Y):
    distances = compute_pairwise_distance(Y)
    inv_distances = 1.0 / (1.0 + distances)
    np.fill_diagonal(inv_distances, 0)
    return np.maximum(inv_distances / np.sum(inv_distances), 1e-12)


def compute_gradients(P, Q, Y):
    n, d = Y.shape
    grad = np.zeros((n, d))
    PQ_diff = P - Q
    inv_dist = 1.0 / (1.0 + compute_pairwise_distance(Y))

    for i in range(n):
        diff = Y[i] - Y
        grad[i] = 4 * np.sum((PQ_diff[:, i] * inv_dist[:, i])[:, None] * diff, axis=0)
    return grad


class BaseTSNE:
    def __init__(self, data, n_components=2, perplexity=30.0,
                 learning_rate=200, n_iter=1000):
        self.X = data
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = 4.0
        self.momentum = 0.5
        self.Y = None
        self.Y_prev = None
        self.P = None
        self.iterations = 0
        self.animation_RPS = 100

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

    def _initialize_embedding(self, X):
        np.random.seed(42)
        self.Y = np.random.randn(X.shape[0], self.n_components) * 1e-4
        self.Y_prev = np.zeros_like(self.Y)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        return X

    def _prepare_optimization(self, X):
        X_normalized = self._initialize_embedding(X)
        distances = compute_pairwise_distance(X_normalized)
        self.P = self._compute_p_matrix(distances)
