import os
import numpy as np
from scipy.linalg import solve, lstsq
import matplotlib.pyplot as plt

def linear_regression():
    print("Running Linear Regression...")
    
    # Create the results folder if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Define the feature matrix X and target vector y
    X = np.array([
        [2100, 3, 20],
        [2500, 4, 15],
        [1800, 2, 30],
        [2200, 3, 25]
    ])
    y = np.array([460, 540, 330, 400])
    
    # Solve using least-squares method
    beta_lstsq, _, _, _ = lstsq(X, y)
    
    # Solve using direct method (X^T X) beta = X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    beta_direct = solve(XtX, Xty)
    
    # Define test cases for comparison
    test_cases = np.array([
        [2100, 3, 20],
        [2400, 3, 20],
        [1800, 2, 30],
        [2500, 4, 15],
        [2000, 3, 10]
    ])
    test_case_labels = ["House 1", "House 2", "House 3", "House 4", "House 5"]
    
    # Predictions for each test case
    predictions_lstsq = test_cases @ beta_lstsq
    predictions_direct = test_cases @ beta_direct
    
    # Save results to a file
    with open("results/linear_regression_results.txt", "w") as file:
        file.write("Regression Coefficients (Least Squares): {}\n".format(beta_lstsq.tolist()))
        file.write("Regression Coefficients (Direct Method): {}\n".format(beta_direct.tolist()))
        file.write("Predictions (Least Squares): {}\n".format(predictions_lstsq.tolist()))
        file.write("Predictions (Direct Method): {}\n".format(predictions_direct.tolist()))
    
    # Create comparison plot
    plt.figure()
    plt.plot(test_case_labels, predictions_lstsq, label="Least Squares", marker='o')
    plt.plot(test_case_labels, predictions_direct, label="Direct Method", marker='x', linestyle='--')
    plt.xlabel("Test Cases")
    plt.ylabel("Predicted Price ($1000s)")
    plt.title("Comparison of Predicted Prices")
    plt.legend()
    plt.savefig("results/linear_regression_comparison_plot.png")  # Save the plot
    plt.show()
    
    print("Linear regression results saved to results/linear_regression_results.txt")
    print("Comparison plot saved to results/linear_regression_comparison_plot.png")
    print()

if __name__ == "__main__":
    linear_regression()
