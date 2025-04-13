from src.kmeans import KMeans
from src.tsne import Tsne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv("data/country/Country-data.csv")
X = data[["child_mort","exports","health","imports","income","inflation","life_expec","total_fer","gdpp"]].to_numpy()
Y = data["country"].to_numpy()

row_norms = np.linalg.norm(X, axis=1, keepdims=True)
row_norms[row_norms == 0] = 1  # avoid division by zero
X = X / row_norms

cluster = KMeans(n_clusters=2, random_state=23, max_iter=300)
n_clusters = cluster.elbow_method(X, max_k=10)
cluster = KMeans(n_clusters=n_clusters, random_state=23, max_iter=300)
cluster.train(X)
Y_predict = cluster.assign_clusters(X)
tsne = Tsne(data=X, n_components=2, perplexity=30, learning_rate=200, n_iter=2000)
Transformed_X = tsne.fit_transform_without_graph(np.vstack((X, cluster.centroids)))
Transformed_centers = Transformed_X[-n_clusters:]
Transformed_X = Transformed_X[:-n_clusters]

display_countries = ["United States", "India", "Germany", "Japan", "China", "Brazil", "Afghanistan", "Chad", "Haiti",
                     "Mexico", "Hungary", "Turkey", "Norway", "Russia", "Jamaica", "Poland", "Italy"]
for i, label in enumerate(Y):
    if label in display_countries:
        plt.text(Transformed_X[i][0] + 0.05, Transformed_X[i][1] + 0.05, label, fontsize=9, fontweight='bold')

plt.scatter(Transformed_X[:, 0], Transformed_X[:, 1], c=Y_predict, cmap='viridis')
plt.scatter(Transformed_centers[:, 0], Transformed_centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.show()


