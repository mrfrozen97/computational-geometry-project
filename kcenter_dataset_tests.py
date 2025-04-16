import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from sklearn.datasets import make_moons
from sklearn.metrics import f1_score, accuracy_score

from src.kcenter import KCenter  # Changed import
from src.tsne import Tsne


def map_clusters_to_labels(y_true, y_pred):
    labels = np.zeros_like(y_pred)
    for cluster in range(10):
        mask = (y_pred == cluster)
        if np.any(mask):
            labels[mask] = mode(y_true[mask])[0]
    return labels


def test_dataset(X, Y, n_clusters, name):
    cluster = KCenter(n_clusters=n_clusters, random_state=27)  # KCenter instance
    cluster.train(X)
    result = json.load(open("results/kcenter/results.json", "r"))  # Changed path
    result[name] = {}
    Y_Predict = map_clusters_to_labels(Y, cluster.labels_)  # Use directly from labels_
    print(f"Accuracy: {accuracy_score(Y, Y_Predict)}")
    print(f"F1-Score: {f1_score(Y, Y_Predict, average='macro')}")
    result[name]["accuracy"] = accuracy_score(Y, Y_Predict)
    result[name]["f1-score"] = f1_score(Y, Y_Predict, average='macro')
    result[name]["max_distance"] = cluster.inertia_  # Store K-center specific metric
    json.dump(result, open("results/kcenter/results.json", "w"), indent=2)

    # Visualization remains similar
    tsne = Tsne(data=X, n_components=2, perplexity=30, learning_rate=200, n_iter=2000)
    Transformed_X = tsne.fit_transform_without_graph(np.vstack((X, cluster.centroids)))
    Transformed_centers = Transformed_X[-n_clusters:]
    Transformed_X = Transformed_X[:-n_clusters]

    plt.scatter(Transformed_X[:, 0], Transformed_X[:, 1], c=Y_Predict, cmap='viridis')
    plt.scatter(Transformed_centers[:, 0], Transformed_centers[:, 1],
                c='red', s=200, alpha=0.75, marker='X')
    plt.savefig(f"results/kcenter/{name}_predicted.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.scatter(Transformed_X[:, 0], Transformed_X[:, 1], c=Y, cmap='viridis')
    plt.scatter(Transformed_centers[:, 0], Transformed_centers[:, 1],
                c='red', s=200, alpha=0.75, marker='X')
    plt.savefig(f"results/kcenter/{name}_actual.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# Synthetic Rings dataset
def generate_multiple_rings(n_rings=3, samples_per_ring=300, noise=0.05):
    X = []
    y = []

    for i in range(n_rings):
        radius = 1 + i * 1.5  # Spread the rings out
        angles = 2 * np.pi * np.random.rand(samples_per_ring)
        x = radius * np.cos(angles) + noise * np.random.randn(samples_per_ring)
        y_ = radius * np.sin(angles) + noise * np.random.randn(samples_per_ring)

        X.append(np.column_stack((x, y_)))
        y.extend([i] * samples_per_ring)

    return np.vstack(X), np.array(y)


# Synthetic Half-moons dataset
def generate_multiple_moons(n_moons=6, samples_per_moon=200, noise=0.05, radius=3.0):
    X_all = []
    y_all = []

    for i in range(n_moons):
        X, _ = make_moons(n_samples=samples_per_moon, noise=noise)

        # Rotate moon
        angle = i * (2 * np.pi / n_moons)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        X_rotated = X @ rotation_matrix.T

        # Move it to a circle
        center_x = radius * np.cos(angle)
        center_y = radius * np.sin(angle)
        X_translated = X_rotated + np.array([center_x, center_y])

        X_all.append(X_translated)
        y_all.append(np.full(samples_per_moon, i))  # label each moon with its index

    return np.vstack(X_all), np.hstack(y_all)


if __name__ == "__main__":
    # Create directory structure first
    import os

    os.makedirs("results/kcenter", exist_ok=True)

    # Initialize empty results file
    with open("results/kcenter/results.json", "w") as f:
        json.dump({}, f)

    # Test with synthetic datasets
    X_rings, y_rings = generate_multiple_rings(n_rings=6, samples_per_ring=200, noise=0.08)
    test_dataset(X_rings, y_rings, n_clusters=10, name="Rings_dataset")

    X_moons, y_moons = generate_multiple_moons()
    test_dataset(X_moons, y_moons, n_clusters=6, name="Half_moons_dataset")
