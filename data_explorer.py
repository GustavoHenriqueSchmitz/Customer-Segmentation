import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("./Mall_Customers.xls")

# Display the first few rows
print("========================== First rows ==============================")
print(df.head())
print("====================================================================")

# Check for missing values
print("=================== Checking for missing values ====================")
print(df.isnull().sum())
print("====================================================================")

# Summary statistics
print("==================== Statisticas Resumidas =========================")
print(df.describe())
print("====================================================================")
