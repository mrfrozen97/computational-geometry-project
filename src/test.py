from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from tsne import Tsne
from tsne_deepseek import TSNE

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

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

