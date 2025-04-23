import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.kcenter import KCenter, elbow_method  # Changed import
from src.tsne import TSNE

if __name__ == "__main__":
    data = pd.read_csv("data/country/Country-data.csv")
    X = data[
        ["child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer",
         "gdpp"]].to_numpy()
    Y = data["country"].to_numpy()

    # Normalization remains the same
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1  # avoid division by zero
    X = X / row_norms

    # K-Center specific initialization
    cluster = KCenter(n_clusters=2, random_state=23)
    n_clusters = elbow_method(X)
    cluster = KCenter(n_clusters=n_clusters, random_state=23)
    cluster.train(X)

    # Directly use labels from training
    Y_predict = cluster.labels_

    # Visualization remains similar
    tsne = TSNE(data=X, n_components=2, perplexity=30, learning_rate=200, n_iter=2000)
    Transformed_X = tsne.fit_transform(np.vstack((X, cluster.centroids)))
    Transformed_centers = Transformed_X[-n_clusters:]
    Transformed_X = Transformed_X[:-n_clusters]

    # Country labeling logic
    display_countries = ["United States", "India", "Germany", "Japan", "China", "Brazil", "Afghanistan", "Chad",
                         "Haiti", "Mexico", "Hungary", "Turkey", "Norway", "Russia", "Jamaica", "Poland", "Italy"]
    for i, label in enumerate(Y):
        if label in display_countries:
            plt.text(Transformed_X[i][0] + 0.05, Transformed_X[i][1] + 0.05, label, fontsize=9, fontweight='bold')

    plt.scatter(Transformed_X[:, 0], Transformed_X[:, 1], c=Y_predict, cmap='viridis')
    plt.scatter(Transformed_centers[:, 0], Transformed_centers[:, 1],
                c='red', s=200, alpha=0.75, marker='X')
    plt.title("Country Clustering with K-Center")
    plt.show()
