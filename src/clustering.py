from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import tensorflow as tf

def Clustering(model, dataset):
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

    return embeddings, clusters