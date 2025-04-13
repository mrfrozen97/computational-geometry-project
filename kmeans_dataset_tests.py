from src.kmeans import KMeans
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score, accuracy_score
from src.tsne import Tsne
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
import json
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.datasets import make_swiss_roll, make_moons, make_circles


def map_clusters_to_labels(y_true, y_pred):
    labels = np.zeros_like(y_pred)
    for cluster in range(10):
        mask = (y_pred == cluster)
        if np.any(mask):
            labels[mask] = mode(y_true[mask])[0]
    return labels

def test_dataset(X, Y, n_clusters, name):
    cluster = KMeans(n_clusters=n_clusters, random_state=27, max_iter=300)
    cluster.train(X)
    result = json.load(open("results/kmeans/results.json", "r"))
    result[name] = {}
    Y_Predict = map_clusters_to_labels(Y, cluster.assign_clusters(X))
    print(f"Accuracy: {accuracy_score(Y, Y_Predict)}")
    print(f"F1-Score: {f1_score(Y, Y_Predict, average='macro')}")
    result[name]["accuracy"] = accuracy_score(Y, Y_Predict)
    result[name]["f1-score"] = f1_score(Y, Y_Predict, average='macro')
    json.dump(result, open("results/kmeans/results.json", "w"), indent=2)
    tsne = Tsne(data=X, n_components=2, perplexity=30, learning_rate=200, n_iter=2000)
    Transformed_X = tsne.fit_transform_without_graph(np.vstack((X, cluster.centroids)))
    Transformed_centers = Transformed_X[-n_clusters:]
    Transformed_X = Transformed_X[:-n_clusters]
    plt.scatter(Transformed_X[:, 0], Transformed_X[:, 1], c=Y_Predict, cmap='viridis')
    plt.scatter(Transformed_centers[:, 0], Transformed_centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.savefig(f"results/kmeans/{name}_predicted.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    plt.scatter(Transformed_X[:, 0], Transformed_X[:, 1], c=Y, cmap='viridis')
    plt.scatter(Transformed_centers[:, 0], Transformed_centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.savefig(f"results/kmeans/{name}_actual.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()




# MNIST dataset
# mnist = fetch_openml('mnist_784', version=1, as_frame=False)
# X, y = mnist["data"], mnist["target"].astype(int)
# # Normalize the data (optional, but often helps)
# X = X / 255.0
#
# # Taking first 1000 due to hardware limitations
# X_small = X[:1000]
# y_small = y[:1000]
# test_dataset(X_small, y_small, n_clusters=10, name="MNIST_dataset")


# Iris dataset
# Load dataset
# iris = load_iris()
# # iris = load_wine()
# X = iris.data
# y = iris.target
# test_dataset(X, y, n_clusters=3, name="IRIS_datset")


# # Wine dataset
# wine = load_wine()
# X = wine.data
# y = wine.target
# test_dataset(X, y, n_clusters=5, name="Wine_datset")


# Breast Cancer dataset
# cancer = load_breast_cancer()
# X = cancer.data
# y = cancer.target
# test_dataset(X, y, n_clusters=2, name="Cancer_datset")


# Synthetic Swiss Roll dataset
# X, y_cont = make_swiss_roll(n_samples=1000, noise=0.1)
# n_classes = 6
# y_class = np.digitize(y_cont, bins=np.linspace(y_cont.min(), y_cont.max(), n_classes))
# test_dataset(X, y_class, n_clusters=n_classes, name="Swiss_Roll_dataset")


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


X_rings, y_rings = generate_multiple_rings(n_rings=6, samples_per_ring=200, noise=0.08)
test_dataset(X_rings, y_rings, n_clusters=10, name="Rings_dataset")


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
            [np.sin(angle),  np.cos(angle)]
        ])
        X_rotated = X @ rotation_matrix.T

        # Move it to a circle
        center_x = radius * np.cos(angle)
        center_y = radius * np.sin(angle)
        X_translated = X_rotated + np.array([center_x, center_y])

        X_all.append(X_translated)
        y_all.append(np.full(samples_per_moon, i))  # label each moon with its index

    return np.vstack(X_all), np.hstack(y_all)

# Generate dataset
X, y = generate_multiple_moons()
test_dataset(X, y, n_clusters=6, name="Half_moons_dataset")
