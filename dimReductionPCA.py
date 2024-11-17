import numpy as np
import matplotlib.pyplot as plt

# Simulate standardized height and weight data
np.random.seed(0)
data = np.random.randn(100, 2)

# Compute the covariance matrix
cov_matrix = np.cov(data, rowvar=False)

# Perform eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Identify the principal components
principal_component = eigenvectors[:, np.argmax(eigenvalues)]

# Project data onto the first principal component
data_1d = data @ principal_component

# Plot original data and 1D projection
plt.scatter(data[:, 0], data[:, 1], label='Original Data', alpha=0.7)
plt.scatter(data_1d, np.zeros_like(data_1d), label='1D Projection', alpha=0.7)
plt.axline((0, 0), slope=principal_component[1]/principal_component[0], color='red', linestyle='--', label='Principal Axis')
plt.legend()
plt.title("Dimensionality Reduction via PCA")
plt.show()
