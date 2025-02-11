import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('Week 2\student_scores.csv', header=0)
print(dataset.head())

scaler = StandardScaler()
dataset[['Hours', 'Scores']] = scaler.fit_transform(dataset[['Hours', 'Scores']])

X = dataset[['Hours']]
y = dataset[['Scores']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE Score: ', rmse)

r2 = r2_score(y_test, y_pred)
print("R2 Score: ", r2)

X_test_orig = scaler.inverse_transform(np.hstack([X_test, np.zeros_like(X_test)]))[:, 0]
y_test_orig = scaler.inverse_transform(np.hstack([np.zeros_like(y_test), y_test]))[:, 1]
y_pred_orig = scaler.inverse_transform(np.hstack([np.zeros_like(y_pred), y_pred]))[:, 1]

plt.scatter(X_test_orig, y_test_orig, color='blue', label='Actual')
plt.plot(X_test_orig, y_pred_orig, color='red', label='Predicted')
plt.xlabel('Study Hours')
plt.ylabel('Scores')
plt.title('Study Hours vs Scores: Actual and Predicted')
plt.legend()
plt.show()