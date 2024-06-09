import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf

def Clustering(model, dataset):
    # Create a model to extract the output of the second last layer (latent space)
    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[2].output)
    embeddings = intermediate_layer_model.predict(dataset)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    score = silhouette_score(embeddings, clusters)
    print(f'Silhouette Score: {score}')
    
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
