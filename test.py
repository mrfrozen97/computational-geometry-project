from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons, fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from src.tsne import Tsne
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


# Load dataset
iris = load_breast_cancer()
#iris = load_wine()
X = iris.data
y = iris.target
print(f"Dimentions: {len(X[0])}")
# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)

# Normalize the data (optional, but often helps)
X = X / 255.0

# If needed, reduce dataset size for speed (e.g., take first 1000)
X = X[:1000]
y = y[:1000]
# Run your t-SNE
tsne = Tsne(data=X, n_components=2, perplexity=30, learning_rate=200, n_iter=2000)
X_embedded = tsne.fit_transform(X, class_Y=y, save_path="./results/tsne/mnist.gif")

# Plot the 2D projection
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis')
plt.title("t-SNE visualization of Iris dataset")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(scatter)
plt.grid(True)
plt.show()


score = silhouette_score(X_embedded, y)
print(f"Silhouette Score: {score:.4f}")

db_score = davies_bouldin_score(X_embedded, y)
print(f"Davies–Bouldin Index: {db_score:.4f}")
