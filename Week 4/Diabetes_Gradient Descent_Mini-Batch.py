import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target.reshape(-1, 1)

print("Dataset Imported as:")
print(type(diabetes))

X = np.hstack([np.ones((X.shape[0], 1)), X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_features = X_train.shape[1]
theta = np.zeros((n_features, 1))
learning_rate = 0.01
epochs = 1000
batch_size = 32

m = len(y_train)

for epoch in range(epochs):
    indices = np.random.permutation(m)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    for start_idx in range(0, m, batch_size):
        end_idx = start_idx + batch_size
        X_batch = X_train_shuffled[start_idx:end_idx]
        y_batch = y_train_shuffled[start_idx:end_idx]

        predictions = np.dot(X_batch, theta)
        error = predictions - y_batch
        gradient = (1 / len(y_batch)) * np.dot(X_batch.T, error)
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