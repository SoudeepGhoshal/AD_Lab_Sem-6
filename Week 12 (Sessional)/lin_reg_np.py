import pandas as pd
import numpy as np

data = pd.read_csv('house_prices.csv', header=0)
x = np.array(data['square_footage'])
y = np.array(data['price'])

x_mean = np.mean(x)
y_mean = np.mean(y)

m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
b = y_mean - m * x_mean

print(f"Slope: {m}, Intercept: {b}")

x_input = float(input('Enter value to predict price: '))
y_pred = m * x_input + b
print(f'Predicted Price: {y_pred}')