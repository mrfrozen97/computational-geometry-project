import numpy as np
from scipy.spatial.distance import pdist, squareform

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200,
                 n_iter=1000, early_exaggeration=12.0, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.random_state = random_state
        self.embedding_ = None

    def _binary_search_perplexity(self, distances, target_perplexity):
        """Binary search to find proper sigma values for each point"""
        n = distances.shape[0]
        sigmas = np.ones(n)
        target_entropy = np.log(target_perplexity)

        for i in range(n):
            dist_row = distances[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]

            sigma_min, sigma_max = 0, None
            sigma = 1.0  # Initial guess

            for _ in range(50):
                # Compute Gaussian kernel with current sigma
                p = np.exp(-dist_row / (2 * sigma**2))
                p_sum = np.sum(p)
                if p_sum == 0:
                    p_sum = 1e-12
                p = p / p_sum

                # Compute entropy
                entropy = -np.sum(p * np.log2(p + 1e-12))

                # Binary search update
                if np.abs(entropy - target_entropy) < 1e-5:
                    break

                if entropy > target_entropy:
                    sigma_min = sigma
                    sigma = sigma * 2 if sigma_max is None else (sigma + sigma_max) / 2
                else:
                    sigma_max = sigma
                    sigma = (sigma_min + sigma) / 2 if sigma_min != 0 else sigma / 2

            # Final probabilities for this point
            p = np.exp(-dist_row / (2 * sigma**2))
            p = p / np.sum(p)
            sigmas[i] = sigma

        return sigmas

    def _compute_p_matrix(self, X):
        """Compute pairwise affinities in high-dimensional space"""
        # Compute squared Euclidean distances
        distances = squareform(pdist(X, 'sqeuclidean'))
        n = X.shape[0]

        # Find optimal sigmas for each point
        sigmas = self._binary_search_perplexity(distances, self.perplexity)

        # Compute pairwise affinities
        P = np.zeros((n, n))
        for i in range(n):
            numerator = np.exp(-distances[i] / (2 * sigmas[i]**2))
            numerator[i] = 0  # Set diagonal to zero
            P[i] = numerator / np.sum(numerator)

        # Symmetrize and normalize
        P = (P + P.T) / (2 * n)
        P = np.maximum(P, 1e-12)  # Ensure no zeros for numerical stability

        return P

    def _compute_q_matrix(self, Y):
        """Compute pairwise affinities in low-dimensional space"""
        distances = squareform(pdist(Y, 'sqeuclidean'))
        inv_distances = 1.0 / (1.0 + distances)
        np.fill_diagonal(inv_distances, 0)
        Q = inv_distances / np.sum(inv_distances)
        Q = np.maximum(Q, 1e-12)
        return Q

    def _compute_gradients(self, P, Q, Y):
        """Compute gradient of KL divergence with respect to Y"""
        n = Y.shape[0]
        grad = np.zeros_like(Y)

        # Compute pairwise differences
        diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]  # shape (n, n, 2)

        # Compute the gradient
        pq_diff = (P - Q)[:, :, np.newaxis]  # shape (n, n, 1)
        inv_dist = 1.0 / (1.0 + squareform(pdist(Y, 'sqeuclidean')))[:, :, np.newaxis]

        grad = 4 * np.sum(pq_diff * inv_dist * diff, axis=1)
        return grad

    def fit_transform(self, X):
        """Fit t-SNE and return the transformed output"""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize variables
        n_samples = X.shape[0]
        Y = np.random.randn(n_samples, self.n_components) * 1e-4
        Y_prev = np.zeros_like(Y)
        gains = np.ones_like(Y)

        # Compute P matrix
        P = self._compute_p_matrix(X)
        P = P * self.early_exaggeration  # Early exaggeration

        # Optimization loop
        for iteration in range(self.n_iter):
            # Compute Q matrix
            Q = self._compute_q_matrix(Y)

            # Stop early exaggeration after 100 iterations
            if iteration == 100:
                P = P / self.early_exaggeration

            # Compute gradient
            grads = self._compute_gradients(P, Q, Y)

            # Update gains (adaptive learning rate)
            gains = (gains + 0.2) * ((grads > 0) != (Y_prev > 0)) + \
                    (gains * 0.8) * ((grads > 0) == (Y_prev > 0))
            gains = np.clip(gains, 0.01, None)

            # Update momentum
            momentum = 0.5 if iteration < 250 else 0.8

            # Update embedding
            Y_update = momentum * Y_prev - self.learning_rate * (gains * grads)
            Y_prev = Y.copy()
            Y += Y_update

            # Center the embedding
            Y -= Y.mean(axis=0)

            # Print progress
            if iteration % 100 == 0 or iteration == self.n_iter - 1:
                kl_divergence = np.sum(P * np.log(P / Q))
                print(f"Iteration {iteration}: KL divergence = {kl_divergence:.4f}")

        self.embedding_ = Y
        return Y
