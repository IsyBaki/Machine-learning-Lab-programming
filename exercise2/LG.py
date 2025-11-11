# linear_regression_lab.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Linear Regression on insurance.csv
# -------------------------------

# (a) Exploratory Analysis
insurance = pd.read_csv("insurance.csv")

print("First few rows:")
print(insurance.head())

print("\nDataset info:")
print(insurance.info())

print("\nBasic statistics:")
print(insurance.describe())

# Plot distributions
numerical_features = ["age", "bmi", "charges"]
for feature in numerical_features:
    plt.figure()
    insurance[feature].hist(bins=20)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

# (b) Data Preprocessing
# One-hot encoding for categorical variables
insurance_encoded = pd.get_dummies(insurance, columns=["sex", "smoker", "region"], drop_first=True)

# Split features and target
X = insurance_encoded.drop("charges", axis=1).values
y = insurance_encoded["charges"].values

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train[:, :3] = scaler.fit_transform(X_train[:, :3])  # Assuming first 3 columns are numerical
X_test[:, :3] = scaler.transform(X_test[:, :3])

# (c) Target transformation
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# (d) Linear Regression from scratch
# Add intercept
X_train_intercept = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_intercept = np.c_[np.ones(X_test.shape[0]), X_test]

# Closed-form solution
beta_hat = np.linalg.pinv(X_train_intercept.T @ X_train_intercept) @ X_train_intercept.T @ y_train_log

# Predictions
y_train_pred = X_train_intercept @ beta_hat
y_test_pred = X_test_intercept @ beta_hat

# (e) Model evaluation: MSE
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse_train = mse(y_train_log, y_train_pred)
mse_test = mse(y_test_log, y_test_pred)

print(f"\nMSE on Training set: {mse_train:.4f}")
print(f"MSE on Test set: {mse_test:.4f}")

# (f) Residual analysis
residuals = y_test_log - y_test_pred
plt.figure()
plt.hist(residuals, bins=20)
plt.title("Histogram of Residuals (Test Set)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# 2. Visualizing Least Squares on toy_data.csv
# -------------------------------

# Load toy dataset
toy = pd.read_csv("toy_data.csv")
X_toy = toy[["x1", "x2"]].values
y_toy = toy["y"].values

# Add intercept
X_toy_intercept = np.c_[np.ones(X_toy.shape[0]), X_toy]

# Compute beta_hat using Least Squares
beta_toy = np.linalg.pinv(X_toy_intercept.T @ X_toy_intercept) @ X_toy_intercept.T @ y_toy
y_toy_pred = X_toy_intercept @ beta_toy

# 3D Visualization
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of actual points
ax.scatter(X_toy[:, 0], X_toy[:, 1], y_toy, color='blue', label='Actual Data')

# Create grid for regression plane
x1_grid, x2_grid = np.meshgrid(np.linspace(X_toy[:, 0].min(), X_toy[:, 0].max(), 20),
                               np.linspace(X_toy[:, 1].min(), X_toy[:, 1].max(), 20))
y_grid = beta_toy[0] + beta_toy[1] * x1_grid + beta_toy[2] * x2_grid

# Plot regression plane
ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5, color='orange', label='Regression Plane')

# Plot residuals
for i in range(len(y_toy)):
    ax.plot([X_toy[i, 0], X_toy[i, 0]], [X_toy[i, 1], X_toy[i, 1]], [y_toy[i], y_toy_pred[i]], color='red')

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
ax.set_title("3D Linear Regression Fit with Residuals")
plt.show()
