import os
import numpy as np
from scipy.linalg import eig, lstsq
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Create the results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# ===========================
# 1. PageRank Algorithm
# ===========================
def pagerank_algorithm():
    print("Running PageRank Algorithm...")
    
    M = np.array([
        [0, 0, 1/2, 0],
        [1/3, 0, 0, 1/2],
        [1/3, 1/2, 0, 1/2],
        [1/3, 1/2, 1/2, 0]
    ])
    
    # Compute the dominant eigenvector
    eigenvalues, eigenvectors = eig(M)
    dominant_eigenvector = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])
    
    # Normalize the eigenvector
    pagerank_scores = dominant_eigenvector / np.sum(dominant_eigenvector)
    
    # Save results to a file
    with open("results/pagerank_results.txt", "w") as file:
        file.write(f"PageRank Scores: {pagerank_scores.tolist()}\n")
        file.write(f"Highest ranked page: {np.argmax(pagerank_scores) + 1}\n")
    
    print("PageRank results saved to results/pagerank_results.txt")
    print()

# ===========================
# 2. Dimensionality Reduction via PCA
# ===========================
def pca_dimensionality_reduction():
    print("Running PCA for Dimensionality Reduction...")
    
    # Simulate standardized height and weight data
    np.random.seed(0)
    data = np.random.randn(100, 2)
    
    # Compute covariance matrix
    cov_matrix = np.cov(data, rowvar=False)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Principal component
    principal_component = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Project data onto principal component
    data_1d = data @ principal_component
    
    # Save plot to results folder
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], label='Original Data', alpha=0.7)
    plt.scatter(data_1d, np.zeros_like(data_1d), label='1D Projection', alpha=0.7)
    plt.axline((0, 0), slope=principal_component[1]/principal_component[0], color='red', linestyle='--', label='Principal Axis')
    plt.legend()
    plt.title("Dimensionality Reduction via PCA")
    plt.savefig("results/pca_plot.png")
    plt.close()
    
    print("PCA plot saved to results/pca_plot.png")
    print()

# ===========================
# 3. Linear Regression via Least Squares
# ===========================
def linear_regression():
    print("Running Linear Regression...")
    
    X = np.array([
        [2100, 3, 20],
        [2500, 4, 15],
        [1800, 2, 30],
        [2200, 3, 25]
    ])
    y = np.array([460, 540, 330, 400])
    
    # Solve for coefficients
    beta, _, _, _ = lstsq(X, y)
    
    # Predict price for a new house
    new_house = np.array([2400, 3, 20])
    predicted_price = new_house @ beta
    
    # Save results to a file
    with open("results/linear_regression_results.txt", "w") as file:
        file.write(f"Regression Coefficients: {beta.tolist()}\n")
        file.write(f"Predicted Price for new house: {predicted_price}\n")
    
    print("Linear regression results saved to results/linear_regression_results.txt")
    print()

# ===========================
# 4. Gradient Descent for Minimizing Loss
# ===========================
def gradient_descent():
    print("Running Gradient Descent...")
    
    # Initialize random matrices
    np.random.seed(0)
    A = np.random.rand(100, 50)
    X = np.random.rand(100, 50)
    
    # Loss function
    def loss_fn(X_flat):
        X_matrix = X_flat.reshape(100, 50)
        return 0.5 * np.sum((X_matrix - A) ** 2)
    
    # Gradient of the loss function
    def grad_fn(X_flat):
        X_matrix = X_flat.reshape(100, 50)
        return (X_matrix - A).flatten()
    
    # Minimize the loss function
    result = minimize(
        fun=loss_fn,
        x0=X.flatten(),
        jac=grad_fn,
        method='CG',
        options={'disp': False, 'maxiter': 1000}
    )
    
    # Save results to a file
    with open("results/gradient_descent_results.txt", "w") as file:
        file.write(f"Final Loss Value: {result.fun}\n")
    
    print("Gradient descent results saved to results/gradient_descent_results.txt")
    print()

# ===========================
# Main Script
# ===========================
if __name__ == "__main__":
    pagerank_algorithm()
    pca_dimensionality_reduction()
    linear_regression()
    gradient_descent()
