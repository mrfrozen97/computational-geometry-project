import numpy as np


class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200, n_iter=1000):
        self.n_components = n_components  # Target dimensionality (default: 2D)
        self.perplexity = perplexity  # Controls balance between local/global structure
        self.learning_rate = learning_rate  # Step size for gradient updates
        self.n_iter = n_iter  # Number of iterations

    def _pairwise_distances(self, X):
        """Compute pairwise squared Euclidean distances for high-dimensional data."""
        sum_X = np.sum(X ** 2, axis=1)
        return -2 * np.dot(X, X.T) + sum_X[:, np.newaxis] + sum_X[np.newaxis, :]

    def _compute_p_matrix(self, distances):
        """Compute P (similarity in high-dimensional space) using a Gaussian kernel."""

        def binary_search_beta(D, target_perp, tol=1e-5, max_iter=50):
            """Binary search to find beta (precision of Gaussian)."""
            beta = np.ones(D.shape[0])
            for i in range(D.shape[0]):
                beta_min, beta_max = -np.inf, np.inf
                for _ in range(max_iter):
                    P_i = np.exp(-D[i] * beta[i])
                    P_i[i] = 0  # Zero self-similarity
                    sum_P = np.sum(P_i)
                    H = np.log(sum_P) + beta[i] * np.sum(D[i] * P_i) / sum_P  # Shannon entropy
                    perp = np.exp(H)
                    if np.abs(perp - target_perp) < tol:
                        break
                    if perp > target_perp:
                        beta_max = beta[i]
                        beta[i] = (beta[i] + beta_min) / 2 if beta_min != -np.inf else beta[i] / 2
                    else:
                        beta_min = beta[i]
                        beta[i] = (beta[i] + beta_max) / 2 if beta_max != np.inf else beta[i] * 2
            return beta

        beta = binary_search_beta(distances, self.perplexity)
        P = np.exp(-distances * beta[:, np.newaxis])
        np.fill_diagonal(P, 0)  # Set self-similarities to 0
        P /= np.sum(P, axis=1, keepdims=True)  # Normalize
        return (P + P.T) / (2 * P.shape[0])  # Symmetrize P

    def _compute_q_matrix(self, Y):
        """Compute Q (similarity in low-dimensional space) using a Student-t kernel."""
        distances = self._pairwise_distances(Y)
        Q = (1 + distances) ** -1
        np.fill_diagonal(Q, 0)
        return Q / np.sum(Q)

    def _compute_gradients(self, P, Q, Y):
        """Compute gradient updates for Y using t-SNE gradient formula."""
        pq_diff = P - Q
        grad = np.zeros_like(Y)
        for i in range(Y.shape[0]):
            grad[i] = np.sum((pq_diff[:, i, np.newaxis] * (Y[i] - Y)), axis=0)
        return grad * 4  # The 4 is a scaling factor used in t-SNE

    def fit_transform(self, X):
        """Perform t-SNE on dataset X."""
        np.random.seed(42)
        Y = np.random.randn(X.shape[0], self.n_components)  # Random initialization

        distances = self._pairwise_distances(X)
        P = self._compute_p_matrix(distances)  # Compute high-dimensional similarities

        for iteration in range(self.n_iter):
            Q = self._compute_q_matrix(Y)  # Compute low-dimensional similarities
            gradients = self._compute_gradients(P, Q, Y)  # Compute gradients
            Y -= self.learning_rate * gradients  # Gradient descent update

            if iteration % 100 == 0:
                loss = np.sum(P * np.log((P + 1e-9) / (Q + 1e-9)))  # Compute KL divergence
                print(f"Iteration {iteration}: KL Divergence = {loss:.4f}")

        return Y
