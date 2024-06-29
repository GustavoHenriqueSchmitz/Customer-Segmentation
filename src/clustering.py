from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import tensorflow as tf

# Activate clustering function
def Clustering(model, dataset):
    # Create a model to extract the output of the second last layer (latent space)
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[2].output)
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

    # Return the second layer embeddings and the formed clusters
    return embeddings, clusters