from preprocess import Preprocess_Data
from train import Train_Model
from clustering import Clustering

dataset = Preprocess_Data()
model = Train_Model(dataset)
Clustering(model, dataset)
