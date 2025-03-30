import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('house_prices.csv', header=0)

sc = StandardScaler()
data[['square_footage', 'price']] = sc.fit_transform(data[['square_footage', 'price']])

X = np.array(data['square_footage']).reshape(-1,1)
y = np.array(data['price']).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_pred, y_test)
rmse = np.sqrt(mse)
print(f'MSE = {mse}')
print(f'RMSE = {rmse}')

plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Predicted Target')
plt.ylabel('Actual Target')
plt.title('Actual vs Predicted Targets')
plt.legend('best')
plt.grid(True)
plt.show()