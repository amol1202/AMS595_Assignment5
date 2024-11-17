import numpy as np
from scipy.linalg import lstsq

# Define the features matrix and target vector
X = np.array([
    [2100, 3, 20],
    [2500, 4, 15],
    [1800, 2, 30],
    [2200, 3, 25]
])
y = np.array([460, 540, 330, 400])

# Solve for coefficients using least squares
beta, _, _, _ = lstsq(X, y)

# Predict the price of a house with given features
new_house = np.array([2400, 3, 20])
predicted_price = new_house @ beta

# Output results
print("Regression Coefficients:", beta)
print("Predicted Price for new house:", predicted_price)
