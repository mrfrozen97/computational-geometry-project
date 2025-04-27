import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.datasets import load_wine, fetch_openml, load_iris, load_breast_cancer, make_swiss_roll
from sklearn.metrics import (accuracy_score, f1_score, silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score,
                             adjusted_mutual_info_score, mutual_info_score)

from kcenter_dataset_tests import generate_multiple_rings, generate_multiple_moons
from src.kcenter import KCenter
from src.kmeans import KMeans
from src.tsne import TSNE


def map_clusters_to_labels(y_true, y_pred):
    labels = np.zeros_like(y_pred)
    for cluster in np.unique(y_pred):
        mask = (y_pred == cluster)
        if np.any(mask):
            labels[mask] = mode(y_true[mask])[0]
    return labels


def plot_comparison(X, Y, Y_pred_kmeans, Y_pred_kcenter, kmeans_centroids, kcenter_centroids, name):
    combined_data = np.vstack((X, kmeans_centroids, kcenter_centroids))

    tsne = TSNE(data=combined_data, n_components=2, perplexity=30, learning_rate=200, n_iter=2000)
    transformed_all = tsne.fit_transform(combined_data)

    X_tsne = transformed_all[:-len(kmeans_centroids) - len(kcenter_centroids)]
    kmeans_centers_tsne = transformed_all[-len(kmeans_centroids) - len(kcenter_centroids):-len(kcenter_centroids)]
    kcenter_centers_tsne = transformed_all[-len(kcenter_centroids):]

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 3, 1)
    scatter_true = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, cmap='viridis')
    plt.scatter(kmeans_centers_tsne[:, 0], kmeans_centers_tsne[:, 1],
                c='red', s=200, marker='X', label='KMeans Centers', alpha=0.75)
    plt.scatter(kcenter_centers_tsne[:, 0], kcenter_centers_tsne[:, 1],
                c='blue', s=200, marker='P', label='KCenter Centers', alpha=0.75)
    plt.title(f"{name} (True Labels with Centers)")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y_pred_kmeans, cmap='viridis')
    plt.scatter(kmeans_centers_tsne[:, 0], kmeans_centers_tsne[:, 1],
                c='red', s=200, marker='X', edgecolor='black', alpha=0.75)
    plt.title(f"{name} (KMeans Clusters)")

    plt.subplot(1, 3, 3)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y_pred_kcenter, cmap='viridis')
    plt.scatter(kcenter_centers_tsne[:, 0], kcenter_centers_tsne[:, 1],
                c='blue', s=200, marker='P', edgecolor='black', alpha=0.75)
    plt.title(f"{name} (KCenter Clusters)")

    plt.tight_layout()
    os.makedirs("results/comparison", exist_ok=True)
    plt.savefig(f"results/comparison/{name}_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def compare_clustering(X, Y, name):
    n_clusters = len(np.unique(Y))
    results = {'Dataset': name}

    kmeans = KMeans(n_clusters=n_clusters, random_state=27, max_iter=300)
    kmeans.train(X)
    Y_pred_kmeans = map_clusters_to_labels(Y, kmeans.labels_)

    results['KMeans Accuracy'] = accuracy_score(Y, Y_pred_kmeans)
    results['KMeans F1 Score'] = f1_score(Y, Y_pred_kmeans, average='macro')
    results['KMeans Inertia'] = kmeans.inertia_
    results['KMeans Silhouette'] = silhouette_score(X, kmeans.labels_)
    results['KMeans Davies-Bouldin'] = davies_bouldin_score(X, kmeans.labels_)
    results['KMeans Calinski-Harabasz'] = calinski_harabasz_score(X, kmeans.labels_)
    results['KMeans Adjusted Rand'] = adjusted_rand_score(Y, kmeans.labels_)
    results['KMeans Normalized MI'] = normalized_mutual_info_score(Y, kmeans.labels_)
    results['KMeans Adjusted MI'] = adjusted_mutual_info_score(Y, kmeans.labels_)
    results['KMeans Mutual Info'] = mutual_info_score(Y, kmeans.labels_)

    kcenter = KCenter(n_clusters=n_clusters, random_state=27)
    kcenter.train(X)
    Y_pred_kcenter = map_clusters_to_labels(Y, kcenter.labels_)

    results['KCenter Accuracy'] = accuracy_score(Y, Y_pred_kcenter)
    results['KCenter F1 Score'] = f1_score(Y, Y_pred_kcenter, average='macro')
    results['KCenter Max Distance'] = kcenter.inertia_
    results['KCenter Silhouette'] = silhouette_score(X, kcenter.labels_)
    results['KCenter Davies-Bouldin'] = davies_bouldin_score(X, kcenter.labels_)
    results['KCenter Calinski-Harabasz'] = calinski_harabasz_score(X, kcenter.labels_)
    results['KCenter Adjusted Rand'] = adjusted_rand_score(Y, kcenter.labels_)
    results['KCenter Normalized MI'] = normalized_mutual_info_score(Y, kcenter.labels_)
    results['KCenter Adjusted MI'] = adjusted_mutual_info_score(Y, kcenter.labels_)
    results['KCenter Mutual Info'] = mutual_info_score(Y, kcenter.labels_)

    plot_comparison(X, Y, kmeans.labels_, kcenter.labels_, kmeans.centroids, kcenter.centroids, name)
    return results


if __name__ == "__main__":
    os.makedirs("results/comparison", exist_ok=True)
    all_results = []

    X_rings, y_rings = generate_multiple_rings()
    all_results.append(compare_clustering(X_rings, y_rings, "Rings"))

    X_moons, y_moons = generate_multiple_moons()
    all_results.append(compare_clustering(X_moons, y_moons, "Moons"))

    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_mnist, y_mnist = mnist.data[:1000] / 255.0, mnist.target[:1000].astype(int)
    all_results.append(compare_clustering(X_mnist, y_mnist, "MNIST"))

    datasets = {
        'Iris': load_iris(),
        'Wine': load_wine(),
        'Breast Cancer': load_breast_cancer(),
        'Swiss Roll': make_swiss_roll(n_samples=1000, noise=0.1)
    }

    for name, data in datasets.items():
        if name == 'Swiss Roll':
            X, y_cont = data
            y = np.digitize(y_cont, bins=np.linspace(y_cont.min(), y_cont.max(), 6))
        else:
            X, y = data.data, data.target
        all_results.append(compare_clustering(X, y, name))

    df = pd.DataFrame(all_results)
    pd.set_option('display.max_columns', None)
    df.to_csv("results/comparison/comparison.csv", index=False)
    print("\nClustering Performance Comparison:")
    print(df.round(3).to_string(index=False))
