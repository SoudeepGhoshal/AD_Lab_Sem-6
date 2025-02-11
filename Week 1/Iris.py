# Write a Python program to demonstrate data preprocessing steps: handling missing values, encoding categorical data, and feature scaling.
# 1. Load a sample dataset using pandas (e.g., Iris and Wine datasets).
# 2. Plot the distribution of a feature using matplotlib.pyplot.hist().
# 3. Create scatter plots to understand relationships between features using seaborn.scatterplot().
# 4. Use a correlation heatmap to find the relationship between multiple features with seaborn.heatmap().

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Loading dataset
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris = pd.read_csv(dataset_url, header=None, names=columns)
print(iris)

# Imputation not needed due to no missing values

# Label Encoding the classes (textual data -> numerical)
label_encoder = LabelEncoder()
iris['class'] = label_encoder.fit_transform(iris['class'])

# Feature scaling
scaler = StandardScaler()
iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(
    iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
)

# Plotting the frequency distribution of Sepal Length 
plt.figure(figsize=(6, 4))
plt.hist(iris['sepal_length'], bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.title('Distribution of Sepal Length', fontsize=16)
plt.xlabel('Sepal Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True)
plt.show()

# Scatter plot relating Sepal Length and Petal Length
plt.figure(figsize=(6, 4))
sns.scatterplot(data=iris, x='sepal_length', y='petal_length', hue='class', palette='Set2')
plt.title('Scatter Plot: Sepal Length vs Petal Length', fontsize=16)
plt.xlabel('Sepal Length', fontsize=12)
plt.ylabel('Petal Length', fontsize=12)
plt.legend(title='Class')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
corr_matrix = iris.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.show()