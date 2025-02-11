import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(dataset_url, header=None, names=columns)

print("Dataset Preview:")
print(data.head())

X = data[['sepal_length', 'sepal_width', 'petal_width']].values
y = data['petal_length'].values.reshape(-1, 1)

X = np.hstack([np.ones((X.shape[0], 1)), X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_features = X_train.shape[1]
theta = np.zeros((n_features, 1))
learning_rate = 0.01
epochs = 1000

m = len(y_train)

for epoch in range(epochs):
    for i in range(m):
        random_index = np.random.randint(0, m)
        x_i = X_train[random_index:random_index + 1]
        y_i = y_train[random_index:random_index + 1]

        prediction = np.dot(x_i, theta)

        error = prediction - y_i

        gradient = x_i.T * error
        theta -= learning_rate * gradient

y_pred = np.dot(X_test, theta)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")