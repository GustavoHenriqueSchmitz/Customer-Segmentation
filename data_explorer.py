# Code that get and analyze information from the dataset
import pandas as pd

# Load the dataset
dataset = pd.read_csv("./Mall_Customers.xls")

print("========================== First rows ==============================")
print(dataset.head())
print("====================================================================")

print("=================== Checking for missing values ====================")
print(dataset.isnull().sum())
print("====================================================================")

print("==================== Statisticas Resumidas =========================")
print(dataset.describe())
print("====================================================================")
