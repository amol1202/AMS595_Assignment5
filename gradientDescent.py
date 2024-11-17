import numpy as np
from scipy.optimize import minimize

# Initialize random matrices X and A
np.random.seed(0)
A = np.random.rand(100, 50)
X = np.random.rand(100, 50)

# Define the loss function
def loss_fn(X_flat):
    X_matrix = X_flat.reshape(100, 50)
    return 0.5 * np.sum((X_matrix - A) ** 2)

# Define the gradient of the loss function
def grad_fn(X_flat):
    X_matrix = X_flat.reshape(100, 50)
    return (X_matrix - A).flatten()

# Minimize using gradient descent
result = minimize(
    fun=loss_fn,
    x0=X.flatten(),
    jac=grad_fn,
    method='CG',
    options={'disp': True, 'maxiter': 1000}
)

# Reshape the optimized matrix
X_optimized = result.x.reshape(100, 50)

# Track and plot the convergence
loss_values = result.fun
print("Final Loss Value:", loss_values)
