from src.preprocess import Preprocess_Data
from src.train import Train_Model
from src.clustering import Clustering

processed_dataset, preprocessor, dataset = Preprocess_Data()
model = Train_Model(processed_dataset)
clustered_data = Clustering(model, processed_dataset, preprocessor, dataset)
