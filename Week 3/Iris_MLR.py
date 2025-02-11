import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris = pd.read_csv(dataset_url, header=None, names=columns)

X = iris[['petal_width', 'petal_length', 'sepal_length']]
y = iris['sepal_width']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-Squared (R²): {r2}")

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(iris['sepal_length'], iris['petal_length'], color='blue', alpha=0.6)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.subplot(2, 2, 2)
plt.scatter(iris['sepal_width'], iris['petal_length'], color='green', alpha=0.6)
plt.title("Sepal Width vs Petal Length")
plt.xlabel("Sepal Width")
plt.ylabel("Petal Length")
plt.subplot(2, 2, 3)
plt.scatter(iris['petal_width'], iris['petal_length'], color='red', alpha=0.6)
plt.title("Petal Width vs Petal Length")
plt.xlabel("Petal Width")
plt.ylabel("Petal Length")
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect Fit")
plt.xlabel("Actual Petal Length")
plt.ylabel("Predicted Petal Length")
plt.title("Actual vs Predicted Petal Length")
plt.legend()
plt.grid(True)
plt.show()