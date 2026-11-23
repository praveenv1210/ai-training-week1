
# =====================================================
# TASK 3: PREDICT LOAN DEFAULT FOR A NEW CUSTOMER
# =====================================================

import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------
# STEP 1: Load the trained model
# -----------------------------------------------------

model = joblib.load("best_bank_loan_model.pkl")

print("\nMODEL LOADED SUCCESSFULLY")


# -----------------------------------------------------
# STEP 2: Get customer details
# -----------------------------------------------------

print("\n========== CUSTOMER DETAILS ==========")

age = float(input("Enter Age: "))

income = float(input("Enter Annual Income: "))

loan_amount = float(input("Enter Loan Amount: "))

credit_score = float(input("Enter Credit Score: "))

employment_years = float(input("Enter Employment Years: "))

existing_debt = float(input("Enter Existing Debt: "))

interest_rate = float(input("Enter Interest Rate: "))

loan_term = float(input("Enter Loan Term: "))


# -----------------------------------------------------
# STEP 3: Create customer DataFrame
# -----------------------------------------------------

customer = pd.DataFrame({
    "age": [age],
    "income": [income],
    "loan_amount": [loan_amount],
    "credit_score": [credit_score],
    "employment_years": [employment_years],
    "existing_debt": [existing_debt],
    "interest_rate": [interest_rate],
    "loan_term": [loan_term]
})


# -----------------------------------------------------
# STEP 4: Feature Engineering
# -----------------------------------------------------

# Income to loan ratio
customer["income_loan_ratio"] = (
    customer["income"] / (customer["loan_amount"] + 1)
)

# Log transformation of loan amount
customer["log_loan_amount"] = np.log1p(
    customer["loan_amount"]
)

# Debt to income ratio
customer["debt_income_ratio"] = (
    customer["existing_debt"] / (customer["income"] + 1)
)

print("\nFEATURE ENGINEERING COMPLETED")

print(customer)


# -----------------------------------------------------
# STEP 5: Arrange features in training order
# -----------------------------------------------------

feature_columns = [
    "age",
    "income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "existing_debt",
    "interest_rate",
    "loan_term",
    "income_loan_ratio",
    "log_loan_amount",
    "debt_income_ratio"
]

customer = customer[feature_columns]


# -----------------------------------------------------
# STEP 6: Predict using Random Forest
# -----------------------------------------------------

prediction = model.predict(customer)

print("\n========== PREDICTION RESULT ==========")

print("Predicted Class:", prediction[0])


# -----------------------------------------------------
# STEP 7: Display prediction
# -----------------------------------------------------

# Target column in the dataset is 'default'
# 0 = No Default
# 1 = Default

if prediction[0] == 0:

    print("\nRESULT: CUSTOMER IS PREDICTED NOT TO DEFAULT")

else:

    print("\nRESULT: CUSTOMER IS PREDICTED TO DEFAULT")


# -----------------------------------------------------
# STEP 8: Display prediction probability
# -----------------------------------------------------

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(customer)[0]

    print("\nPREDICTION PROBABILITY:")

    for class_label, probability in zip(
        model.classes_, probabilities
    ):

        print(
            f"Class {class_label}: {probability * 100:.2f}%"
        )
print("\nMODEL VERIFICATION")

print("Model Type:", type(model))

print("Expected Features:", model.n_features_in_)

print("Feature Names:")

if hasattr(model, "feature_names_in_"):
    print(model.feature_names_in_)