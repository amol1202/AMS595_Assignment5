import numpy as np
from scipy.linalg import eig

# Define the stochastic matrix M
M = np.array([
    [0, 0, 1/2, 0],
    [1/3, 0, 0, 1/2],
    [1/3, 1/2, 0, 1/2],
    [1/3, 1/2, 1/2, 0]
])

# Compute the dominant eigenvector using eig
eigenvalues, eigenvectors = eig(M)
dominant_eigenvector = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])

# Normalize the eigenvector so its sum is 1
pagerank_scores = dominant_eigenvector / np.sum(dominant_eigenvector)

# Output the results
print("PageRank Scores:", pagerank_scores)
print("Highest ranked page:", np.argmax(pagerank_scores) + 1)
