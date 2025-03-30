import numpy as np

data = np.array([
    [4, 11],
    [8, 4],
    [13, 5],
    [7, 14]
])

mean = np.mean(data, axis=0)
data_centered = data - mean

cov_matrix = np.cov(data_centered, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

k = 1
selected_eigenvectors = eigenvectors[:, :k]

reduced_data = np.dot(data_centered, selected_eigenvectors)

print("Projected Reduced Dataset:")
print(reduced_data)