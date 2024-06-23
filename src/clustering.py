from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import tensorflow as tf

def Clustering(model, dataset, preprocessor, original_data):
    # Create a model to extract the output of the second last layer (latent space)
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[2].output)
    embeddings = intermediate_layer_model.predict(dataset)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    score = silhouette_score(embeddings, clusters)
    db_index = davies_bouldin_score(embeddings, clusters)
    ch_index = calinski_harabasz_score(embeddings, clusters)
    print(f'Silhouette Score: {score}')
    print(f'Davies-Bouldin Index: {db_index}')
    print(f'Calinski-Harabasz Index: {ch_index}')
    
    # Visualize clusters
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k')
    plt.title('Customer Segments')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()

    # Ensure the cluster labels match the original dataset
    original_data['Cluster'] = clusters

    # Exclude non-numeric columns for mean calculation
    numeric_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    cluster_summary_numeric = original_data.groupby('Cluster')[numeric_features].mean()
    cluster_summary_non_numeric = original_data.groupby('Cluster')['Genre'].agg(lambda x: x.mode()[0])

    # Combine numeric and non-numeric summaries
    cluster_summary = pd.concat([cluster_summary_numeric, cluster_summary_non_numeric], axis=1)
    print(cluster_summary)

    return original_data
