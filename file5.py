import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset

df = pd.read_csv('PositionSalaries.csv')
print(df.head())
df.info()                                                                                                                                                                                        


# Features and target
X = df.iloc[:, 1:2].values   # Level
y = df.iloc[:, 2].values     # Salary

# Train Random Forest
regressor = RandomForestRegressor(n_estimators=20, random_state=10)
regressor.fit(X, y)

# Predictions
X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
y_pred = regressor.predict(X)

# Metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f'MSE: {mse:.2f}, R²: {r2:.2f}')

# Plot regression curve
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X_grid, regressor.predict(X_grid), color='green', label="Random Forest Prediction")
plt.title("Random Forest Regression Results")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Visualize one tree
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
plot_tree(regressor.estimators_[0], feature_names=['Level'], filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree from Random Forest")
plt.show()