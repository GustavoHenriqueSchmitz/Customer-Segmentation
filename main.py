from src.preprocess import Preprocess_Data
from src.autoencoder import Autoencoder
from src.clustering import Clustering
from src.results import Results

processed_dataset, dataset = Preprocess_Data()
model = Autoencoder(processed_dataset)
embeddings, clusters = Clustering(model, processed_dataset)
dataset = Results(dataset, embeddings, clusters)

