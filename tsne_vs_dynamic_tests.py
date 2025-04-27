import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine, load_iris, load_breast_cancer, make_swiss_roll, fetch_openml
from sklearn.metrics import silhouette_score, davies_bouldin_score

from kcenter_dataset_tests import generate_multiple_rings, generate_multiple_moons
from src.tsne import TSNE
from src.tsne_dynamic import TSNEDynamic


def compare_visualizations(embedding_standard, embedding_dynamic, y, dataset_name):
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(embedding_standard[:, 0], embedding_standard[:, 1], c=y, cmap='viridis')
    plt.title("Standard t-SNE")
    plt.colorbar(scatter1)

    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(embedding_dynamic[:, 0], embedding_dynamic[:, 1], c=y, cmap='viridis')
    plt.title("Dynamic Convergence t-SNE")
    plt.colorbar(scatter2)

    plt.tight_layout()
    plt.savefig(f"results/tsne_comparison/{dataset_name}_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def run_tsne_methods(X, y, dataset_name):
    results = {"Dataset": dataset_name}

    start = time.time()
    tsne = TSNE(data=X, n_components=2, perplexity=30, learning_rate=200, n_iter=2000)
    X_standard = tsne.fit_transform(X)
    end = time.time()
    results.update({
        "Standard Silhouette": silhouette_score(X_standard, y),
        "Standard DBI": davies_bouldin_score(X_standard, y),
        "Standard Iter": tsne.iterations,
        "Standard Time (s)": end - start
    })

    start = time.time()
    tsne_dyn = TSNEDynamic(
        data=X, n_components=2, perplexity=30, learning_rate=200,
        target_score=0.9, check_interval=20, n_iter=2000, tolerance_percentage=2
    )
    X_dynamic = tsne_dyn.fit_transform(X, labels=y)
    end = time.time()
    results.update({
        "Dynamic Silhouette": silhouette_score(X_dynamic, y),
        "Dynamic DBI": davies_bouldin_score(X_dynamic, y),
        "Dynamic Iter": tsne_dyn.iterations,
        "Dynamic Time (s)": end - start
    })

    compare_visualizations(X_standard, X_dynamic, y, dataset_name)

    return results


if __name__ == "__main__":
    os.makedirs("results/tsne_comparison", exist_ok=True)

    all_results = []
    all_results.append(run_tsne_methods(*generate_multiple_rings(n_rings=6, samples_per_ring=200, noise=0.08), "Rings"))
    all_results.append(run_tsne_methods(*generate_multiple_moons(), "Moons"))

    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_mnist, y_mnist = mnist["data"][:1000] / 255.0, mnist["target"][:1000].astype(int)
    all_results.append(run_tsne_methods(X_mnist, y_mnist, "MNIST (1k)"))

    datasets = {
        "Iris": load_iris(),
        "Wine": load_wine(),
        "Breast Cancer": load_breast_cancer(),
        "Swiss Roll": make_swiss_roll(n_samples=1000, noise=0.1)
    }

    for name, data in datasets.items():
        if name == "Swiss Roll":
            X, y_cont = data
            y_class = np.digitize(y_cont, bins=np.linspace(y_cont.min(), y_cont.max(), 6))
            all_results.append(run_tsne_methods(X, y_class, name))
        else:
            X, y = data.data, data.target
            all_results.append(run_tsne_methods(X, y, name))

    df = pd.DataFrame(all_results)
    df.to_csv("results/tsne_comparison/tsne_dynamic_tests.csv", index=False)
    print(df.to_string(index=False))
