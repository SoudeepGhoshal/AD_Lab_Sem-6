import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the Wine dataset
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
columns = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids',
           'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
wine = pd.read_csv(dataset_url, header=None, names=columns)
print(wine.head())

# Handling missing values
print(wine.isnull().sum()) #Print the number of missing values
# No imputation needed due to no missing value

# Feature scaling
scaler = StandardScaler()
wine.iloc[:, 1:] = scaler.fit_transform(wine.iloc[:, 1:])

# Plotting the distribution of Alcohol
plt.figure(figsize=(6, 4))
plt.hist(wine['alcohol'], bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.title('Distribution of Alcohol', fontsize=16)
plt.xlabel('Alcohol (scaled)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True)
plt.show()

# Scatter plot of Alcohol vs Color Intensity
plt.figure(figsize=(6, 4))
sns.scatterplot(data=wine, x='alcohol', y='color_intensity', hue='class', palette='Set2')
plt.title('Scatter Plot: Alcohol vs Color Intensity', fontsize=16)
plt.xlabel('Alcohol (scaled)', fontsize=12)
plt.ylabel('Color Intensity (scaled)', fontsize=12)
plt.legend(title='Class')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
corr_matrix = wine.iloc[:, 1:].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.show()
