from src.preprocess import Preprocess_Data
from src.train import Train_Model
from src.clustering import Clustering

dataset = Preprocess_Data()
model = Train_Model(dataset)
Clustering(model, dataset)
