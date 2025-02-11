import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('Week 2\height-weight.csv', header=0)
print(dataset.head(5))

dataset_enc = pd.get_dummies(dataset, columns=['Gender'], drop_first=True)

scaler = StandardScaler()
dataset_enc[['Height', 'Weight']] = scaler.fit_transform(dataset_enc[['Height', 'Weight']])
print(dataset_enc)

X = dataset_enc[['Height', 'Gender_Male']]
y = dataset_enc[['Weight']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE Score: ', rmse)

r2 = r2_score(y_test, y_pred)
print("R2 Score: ", r2)

plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Weight (Standardized)')
plt.ylabel('Predicted Weight (Standardized)')
plt.title('Actual vs Predicted Weight')
plt.show()