from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import tensorflow as tf

def Clustering(model, dataset):
    """
    Generate customer clusters based on the latent space representation of the trained autoencoder model.

    Args:
        model: Trained autoencoder model
        dataset: Loaded preprocessed dataset

    Returns:
        embeddings: Latent space representations of the dataset obtained from the latent layer of the autoencoder.
        clusters: The generated clusters by the K-means
    """
    # Extract the latent space layer by name
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("latent_space").output)
    embeddings = intermediate_layer_model.predict(dataset)
    
    # Define and execute the K-means algorithm
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Calculate and log the clustering performance measures
    sl_score = silhouette_score(embeddings, clusters)
    db_index = davies_bouldin_score(embeddings, clusters)
    ch_index = calinski_harabasz_score(embeddings, clusters)
    print("======================== Clustering Performance Measures ========================")
    print(f'Silhouette Score: {sl_score}')
    print(f'Davies-Bouldin Index: {db_index}')
    print(f'Calinski-Harabasz Index: {ch_index}')
    print("=================================================================================")

    # Return embeddings and clusters
    return embeddings, clusters
