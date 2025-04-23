import time

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.tsne import TSNE
from src.tsne_dynamic import TSNEDynamic


def compare_visualizations(embedding_standard, embedding_dynamic, y):
    """Helper function to plot both results side-by-side"""
    plt.figure(figsize=(16, 6))

    # Standard t-SNE plot
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(embedding_standard[:, 0], embedding_standard[:, 1], c=y, cmap='viridis')
    plt.title("Standard t-SNE")
    plt.colorbar(scatter1)

    # Dynamic t-SNE plot
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(embedding_dynamic[:, 0], embedding_dynamic[:, 1], c=y, cmap='viridis')
    plt.title("Dynamic Convergence t-SNE")
    plt.colorbar(scatter2)

    plt.tight_layout()
    plt.show()


def test_tsne(X, y):
    """Test standard t-SNE implementation"""
    print("\n=== Testing Standard t-SNE ===")
    tsne = TSNE(data=X, n_components=2, perplexity=30, learning_rate=200, n_iter=2000)
    X_embedded = tsne.fit_transform(X)

    # Calculate metrics
    sil_score = silhouette_score(X_embedded, y)
    db_score = davies_bouldin_score(X_embedded, y)

    print(f"Standard t-SNE Results:")
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Davies-Bouldin Index: {db_score:.4f}")

    return X_embedded, tsne.iterations


def test_tsne_dynamic(X, y):
    """Test dynamic convergence t-SNE implementation"""
    print("\n=== Testing Dynamic t-SNE ===")
    tsne = TSNEDynamic(
        data=X,
        n_components=2,
        perplexity=30,
        learning_rate=200,
        target_score=0.9,
        check_interval=20,
        n_iter=2000,
        tolerance_percentage=2
    )

    X_embedded = tsne.fit_transform(X, labels=y)

    # Calculate metrics
    sil_score = silhouette_score(X_embedded, y)
    db_score = davies_bouldin_score(X_embedded, y)

    print(f"Dynamic t-SNE Results:")
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Davies-Bouldin Index: {db_score:.4f}")
    print(f"Final Cluster Score: {tsne.score:.4f}")

    return X_embedded, tsne.iterations


if __name__ == "__main__":
    # Load dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_big, y = mnist["data"], mnist["target"].astype(int)
    # Normalize the data (optional, but often helps)
    X_big = X_big / 255.0

    # # Taking first 1000 due to hardware limitations
    X = X_big[:1000]
    y = y[:1000]
    print(f"Dataset dimensions: {X.shape[1]}")

    # Run both tests with timing
    start_standard = time.time()
    standard_embed, iterations_standard = test_tsne(X, y)
    end_standard = time.time()
    standard_duration = end_standard - start_standard

    start_dynamic = time.time()
    dynamic_embed, iterations_dynamic = test_tsne_dynamic(X, y)
    end_dynamic = time.time()
    dynamic_duration = end_dynamic - start_dynamic

    # Compare visualizations
    compare_visualizations(standard_embed, dynamic_embed, y)

    # Compare runtime characteristics
    print("\n=== Comparative Summary ===")
    print(f"Standard t-SNE iterations: {iterations_standard}")
    print(f"Standard t-SNE execution time: {standard_duration:.2f} seconds")
    print(f"Dynamic t-SNE iterations: {iterations_dynamic}")
    print(f"Dynamic t-SNE execution time: {dynamic_duration:.2f} seconds")
