from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from src.tsne import Tsne

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)

# Normalize the data (optional, but often helps)
X = X / 255.0

# If needed, reduce dataset size for speed (e.g., take first 1000)
X_small = X[:400]
y_small = y[:400]
color_y = []
colors = ["blue", "red", "orange", "yellow", "green", "black", "cyan", "pink", "grey", "purple"]
print(y_small)
for i in range(len(y_small)):
    temp = y_small[i]
    color_y.append(colors[temp])
print(color_y)
# Run your t-SNE
tsne = Tsne(data=X, n_components=2, perplexity=30, learning_rate=100, n_iter=20000)
X_embedded = tsne.fit_transform(X_small, class_Y=color_y)

# Plot the 2D projection
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis')
plt.title("t-SNE visualization of Iris dataset")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(scatter)
plt.grid(True)
plt.show()
