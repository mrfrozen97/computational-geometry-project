from sklearn.datasets import load_iris, load_wine
import matplotlib.pyplot as plt
import numpy as np
from src.tsne import Tsne
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


# Load dataset
#iris = load_iris()
iris = load_wine()
X = iris.data
y = iris.target
print(f"Dimentions: {len(X[0])}")

# Run your t-SNE
tsne = Tsne(data=X, n_components=2, perplexity=10, learning_rate=30, n_iter=20000)
X_embedded = tsne.fit_transform(X, class_Y=y)

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
