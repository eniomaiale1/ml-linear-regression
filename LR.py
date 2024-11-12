import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import make_pipeline

# Load data from 'data.csv' file
data = pd.read_csv('data.csv')
X = data[['x']].values
y = data['y'].values

# Define degrees for models
degree = 25   # High degree for overfitting and robust models

# --------- Overfitted Model (High-Degree Polynomial Regression) ---------

# Generate polynomial features and fit a linear regression model
poly_overfit = PolynomialFeatures(degree=degree)
X_poly_overfit = poly_overfit.fit_transform(X)
overfit_model = LinearRegression()
overfit_model.fit(X_poly_overfit, y)

# Predict values for plotting the overfitted model
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly_overfit = poly_overfit.transform(X_range)
y_pred_overfit = overfit_model.predict(X_range_poly_overfit)

# --------- Robust Model (Moderate LASSO Regularization) ---------

# Set up LASSO with moderate regularization
alpha_robust = 0.001
lasso_robust = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=alpha_robust, max_iter=10000))
lasso_robust.fit(X, y)

# Predict values for plotting the robust model
y_pred_robust = lasso_robust.predict(X_range)

# --------- Underfitted Model (High LASSO Regularization, nearly linear) ---------

# Set up LASSO with strong regularization to achieve an underfitted (almost linear) model
alpha_underfit = 10000  # Increase alpha to make the model almost linear
lasso_underfit = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=alpha_underfit, max_iter=10000))
lasso_underfit.fit(X, y)

# Predict values for plotting the underfitted model
y_pred_underfit = lasso_underfit.predict(X_range)

# --------- Plot All Models ---------

plt.figure(figsize=(10, 10))

# Plot original data points
plt.scatter(X, y, color='blue', label='Data')

# Plot the overfitted model
plt.plot(X_range, y_pred_overfit, color='orange', linestyle='--', label=f'Overfitted Model (Degree {degree})')

# Plot the robust model
plt.plot(X_range, y_pred_robust, color='red', label=f'Robust Model with LASSO (Alpha={alpha_robust})')

# Plot the underfitted model
plt.plot(X_range, y_pred_underfit, color='green', linestyle=':', label=f'Underfitted Model with LASSO (Alpha={alpha_underfit})')

# Labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Overfitted, Robust, and Underfitted Models')
plt.legend()
plt.show()

# Output coefficients for each model to analyze term reduction
poly_features = PolynomialFeatures(degree)
X_poly = poly_features.fit_transform(X)

# Overfitted model coefficients
print("Overfitted Model Coefficients:", overfit_model.coef_)

# Robust model coefficients
lasso_robust_model = Lasso(alpha=alpha_robust, max_iter=10000)
lasso_robust_model.fit(X_poly, y)
print("Robust Model Coefficients (LASSO, moderate):", lasso_robust_model.coef_)

# Underfitted model coefficients
lasso_underfit_model = Lasso(alpha=alpha_underfit, max_iter=10000)
lasso_underfit_model.fit(X_poly, y)
print("Underfitted Model Coefficients (LASSO, strong):", lasso_underfit_model.coef_)
