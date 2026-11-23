
# ==========================================
# TASK 2: PREPROCESSING, FEATURES AND PIPELINE
# ==========================================

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("bank_loan_dataset.csv")

# Display dataset information
print("\nDATASET INFORMATION")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN INFORMATION:")
print(df.info())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDATA TYPES:")
print(df.dtypes)

print(df.info())


# TASK 2 - STEP 1: Check Missing Values

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# Separate features and target
X = df.drop("default", axis=1)
y = df["default"]

print("\nFEATURE SHAPE:", X.shape)
print("TARGET SHAPE:", y.shape)


# TASK 2 - STEP 2: Identify Column Types

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identify numerical and categorical columns
numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNUMERIC COLUMNS:")
print(numeric_columns)

print("\nCATEGORICAL COLUMNS:")
print(categorical_columns)


# TASK 2 - STEP 3: Preprocessing Pipelines

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

# Numerical preprocessing using SimpleImputer
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Numerical preprocessing using KNNImputer
knn_pipeline = Pipeline(steps=[
    ("imputer", KNNImputer(n_neighbors=5)),
    ("scaler", StandardScaler())
])

# Test both imputers
simple_result = numeric_pipeline.fit_transform(X)
knn_result = knn_pipeline.fit_transform(X)

print("\nSIMPLE IMPUTER RESULT:", simple_result.shape)
print("KNN IMPUTER RESULT:", knn_result.shape)

print("\nPREPROCESSING COMPLETED")


# TASK 2 - STEP 4: FEATURE ENGINEERING

import numpy as np

# 1. Income to loan ratio
X["income_loan_ratio"] = (
    X["income"] / (X["loan_amount"] + 1)
)

# 2. Log transformation of loan amount
X["log_loan_amount"] = np.log1p(X["loan_amount"])

# 3. Debt to income ratio
X["debt_income_ratio"] = (
    X["existing_debt"] / (X["income"] + 1)
)

print("\nFEATURE ENGINEERING RESULTS")

print(X[[
    "income_loan_ratio",
    "log_loan_amount",
    "debt_income_ratio"
]].head())

print("\nNEW FEATURE SHAPE:", X.shape)


# TASK 2 - STEP 5: Train-Test Split

from sklearn.model_selection import train_test_split

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA SHAPE:", X_train.shape)
print("TESTING DATA SHAPE:", X_test.shape)
print("TRAINING TARGET SHAPE:", y_train.shape)
print("TESTING TARGET SHAPE:", y_test.shape)


# TASK 2 - STEP 6: Model Training

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create machine learning pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETED")


# TASK 2 - STEP 7: Model Evaluation

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:", accuracy)

# Confusion matrix
print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

# Classification report
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))


# TASK 2 - STEP 8: RANDOM FOREST MODEL

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Create Random Forest pipeline
rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Train Random Forest
rf_model.fit(X_train, y_train)

# Predict test data
rf_pred = rf_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRANDOM FOREST RESULTS")
print("Accuracy:", rf_accuracy)

print("\nMODEL COMPARISON")
print("Logistic Regression:", accuracy)
print("Random Forest:", rf_accuracy)


# TASK 2 - STEP 9: SAVE BEST MODEL

import pickle

# Save Random Forest model
with open("best_bank_loan_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

print("\nMODEL SAVED SUCCESSFULLY")

# Load saved model
with open("best_bank_loan_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

print("MODEL LOADED SUCCESSFULLY")

# Test loaded model
loaded_predictions = loaded_model.predict(X_test)

print("LOADED MODEL ACCURACY:",
      accuracy_score(y_test, loaded_predictions))
