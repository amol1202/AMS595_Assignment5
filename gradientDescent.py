import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Initialize random matrices
np.random.seed(0)
A = np.random.rand(100, 50)
X = np.random.rand(100, 50)

# Track loss values
loss_values = []

# Define the loss function
def loss_fn(X_flat):
    X_matrix = X_flat.reshape(100, 50)
    loss = 0.5 * np.sum((X_matrix - A) ** 2)
    loss_values.append(loss)  # Track the loss at each iteration
    return loss

# Define the gradient of the loss function
def grad_fn(X_flat):
    X_matrix = X_flat.reshape(100, 50)
    return (X_matrix - A).flatten()

# Minimize the loss function
result = minimize(
    fun=loss_fn,
    x0=X.flatten(),
    jac=grad_fn,
    method='CG',
    options={'disp': True, 'maxiter': 1000}
)

# Plot the loss values
plt.figure()
plt.plot(loss_values, label="Loss Value")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Value Over Iterations")
plt.legend()
plt.savefig("results/loss_plot.png")  # Save the plot
plt.show()
