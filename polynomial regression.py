import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate some random data for demonstration
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 1 + 2 * X + X**2 + np.random.rand(100, 1)

# Create polynomial features (in this case, up to X^2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Create and train a polynomial regression model
poly_regression = LinearRegression()
poly_regression.fit(X_poly, y)

# Make predictions
y_pred = poly_regression.predict(X_poly)

# Calculate mean squared error (MSE) to evaluate the model
mse = mean_squared_error(y, y_pred)

# Plot the original data and the regression curve
plt.scatter(X, y, label='Data')
plt.plot(X, y_pred, color='red', label='Polynomial Regression')
plt.legend()
plt.title(f'Polynomial Regression (MSE={mse:.2f})')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
