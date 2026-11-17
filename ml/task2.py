from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


from sklearn.datasets import load_diabetes


# Load Diabetes dataset
diabetes = load_diabetes()


X = diabetes.data
y = diabetes.target


# Display dataset
df = pd.DataFrame(
    X,
    columns=diabetes.feature_names
)


df['target'] = y



print("First 5 Records:")
print(df.head())


print("\nDataset Shape:", df.shape)



# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# Create Linear Regression model
model = LinearRegression()


# Train the model
model.fit(X_train, y_train)


# Predict test data
y_pred = model.predict(X_test)


print("\nLinear Regression Model Trained Successfully!")
print("First 5 Actual Values:", y_test[:5])
print("First 5 Predicted Values:", y_pred[:5])




# Calculate Regression Metrics


# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)


# Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


# R2 Score
r2 = r2_score(y_test, y_pred)


print("\nLinear Regression Metrics")
print("-------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))



# Baseline Model


# Calculate mean of training target
mean_value = np.mean(y_train)


# Predict the same mean for all test records
baseline_pred = np.full(len(y_test), mean_value)


# Calculate baseline metrics
baseline_mae = mean_absolute_error(
    y_test, baseline_pred
)


baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_pred)
)


baseline_r2 = r2_score(
    y_test, baseline_pred
)


print("\nBaseline Model Metrics")
print("----------------------")
print("MAE:", round(baseline_mae, 2))
print("RMSE:", round(baseline_rmse, 2))
print("R2 Score:", round(baseline_r2, 2))



# Compare Baseline and Linear Regression


comparison = pd.DataFrame({
    'Model': ['Baseline', 'Linear Regression'],
    'MAE': [baseline_mae, mae],
    'RMSE': [baseline_rmse, rmse],
    'R2': [baseline_r2, r2]
})


print("\nModel Comparison:")
print(comparison.round(2).to_string(index=False))