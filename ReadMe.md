# Dataset Names
* Word2Vec or GloVe Embeddings
Word2Vec and GloVe are popular word embedding models that map words to continuous vector representations in a high-dimensional space (e.g., 100, 300 dimensions). These embeddings capture semantic relationships between words, making them ideal for visualizing with t-SNE.
Datasets:
Google Word2Vec Pretrained Embeddings
GloVe Pretrained Embeddings
FastText Embeddings: Pretrained FastText Embeddings
You can load these embeddings and use t-SNE to visualize how words with similar meanings are clustered together in a 2D or 3D space.

* MNIST
The MNIST dataset consists of handwritten digits (0-9) and is often used for image classification tasks. Although it’s relatively simple, t-SNE can be used to project high-dimensional pixel data into lower dimensions and visualize the clustering of the digits.
Dataset: MNIST Dataset

* UCI Machine Learning Repository
The UCI Repository provides a wide range of datasets, from text and classification to image data. Some of them are suitable for dimensionality reduction tasks like t-SNE.
Examples:
Iris Dataset – A simple 4-dimensional dataset with 3 classes of flowers.
Wine Dataset – A dataset with 13 features describing wine types.
Digits Dataset – A set of 8x8 pixel images of handwritten digits.

* Fashion MNIST
Fashion MNIST is similar to MNIST, but the images represent articles of clothing like t-shirts, pants, and shoes. It’s a good alternative for testing image embeddings with t-SNE.

* 20 Newsgroups
20 Newsgroups is a collection of approximately 20,000 newsgroup documents, partitioned across 20 categories. It’s ideal for testing t-SNE in a text classification scenario where the dimensionality reduction of the text data could reveal interesting patterns.

* Cora / Citeseer / PubMed (Citation Networks)
These datasets represent citation networks where each node is a paper, and the edges represent citations between papers. The papers are categorized, and t-SNE can be used to visualize the relationships and clustering between paper categories based on features extracted from the citation graph.

* ImageNet (Feature Embeddings)
ImageNet contains millions of images across thousands of categories. You can use pre-trained models (like ResNet, VGG, etc.) to extract feature vectors for the images, and then apply t-SNE to visualize these features in 2D.


* Ways to measure quality of TSNE clusters

1. Silhouette Score
Measures how similar a point is to its own cluster vs. others.
Values range from -1 to 1:
   * ~1: Well-separated, cohesive clusters.
   * 0: Overlapping clusters.
   * <0: Likely incorrect clustering.

2. Davies–Bouldin Index (DBI)
   * Measures intra-cluster spread vs. inter-cluster distance.
   * Lower is better.
   * Works well for t-SNE output to gauge tightness + separation.

4. Calinski-Harabasz Index
   * Measures between-cluster dispersion / within-cluster dispersion.
   * Higher = better separation.
   * Especially useful when you know true cluster labels.

# Kmeans Implementation
* We use Elbow method to determine the number of cluster centers.
* We assign label to cluster centers based on voting method, which means that the label of cluster is majority label of elements in the cluster.
* Normalize data before use so that each dimension effects the cluster equally


# References

