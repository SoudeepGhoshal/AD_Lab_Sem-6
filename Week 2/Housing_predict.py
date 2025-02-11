import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('Week 2\Housing.csv', header=0)
print(dataset.head())

categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                    'airconditioning', 'prefarea', 'furnishingstatus']
dataset_encoded = pd.get_dummies(dataset, columns=categorical_cols, drop_first=True)
print(dataset_encoded.head())

scaler = StandardScaler()
dataset_encoded[['area']] = scaler.fit_transform(dataset_encoded[['area']])

X = dataset_encoded.drop('price', axis=1)
y = dataset_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
min_price = y.min()
max_price = y.max()
normalized_rmse = abs((rmse - min_price) / (max_price - min_price))
print('RMSE Score: ', normalized_rmse)

r2 = r2_score(y_test, y_pred)
print("R2 Score: ", r2)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.show()