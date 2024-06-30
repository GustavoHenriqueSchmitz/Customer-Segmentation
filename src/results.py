import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

def Results(dataset, embeddings, clusters):
    """
    Show the clustering results.
    Show a plot to visualize the clusters and the generated clusters

    Args:
        dataset: The original dataset
        embeddings: Latent Space representations of the dataset obtained from the second last layer of the autoencoder. 
        clusters: Generated clusters by the K-means
    """
    # Make the cluster labels match the original dataset
    dataset['Cluster'] = clusters

    # Exclude non-numeric columns for mean calculation
    numeric_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    cluster_summary_numeric = dataset.groupby('Cluster')[numeric_features].mean()
    cluster_summary_non_numeric = dataset.groupby('Cluster')['Genre'].agg(lambda x: x.mode()[0])

    # Combine numeric and non-numeric summaries
    cluster_summary = pd.concat([cluster_summary_numeric, cluster_summary_non_numeric], axis=1)
    print("================================ Clusters Summary ================================")
    print(cluster_summary)
    print("==================================================================================")
    
    # Reduce the embeddings to 2 dimensions for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(embeddings)
    
    # Generate a plot to better visualize clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k')
    plt.title('Customers Clusters Distribution')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.show()
