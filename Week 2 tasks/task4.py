
# ============================================================
# TASK 4 - EVALUATION AND TUNING
# BANK LOAN DEFAULT PREDICTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

print("\n========== TASK 4 STARTED ==========")

df = pd.read_csv("bank_loan_dataset.csv")

print("\nDATASET INFORMATION")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nSample Data:")
print(df.head())


# ============================================================
# STEP 2: DEFINE FEATURES AND TARGET
# ============================================================

TARGET = "default"

FEATURES = [
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
TARGET="default"

missing_columns = [
    col for col in FEATURES + [TARGET]
    if col not in df.columns
]
# ==========================================
# FEATURE ENGINEERING
# ==========================================

import numpy as np

# Income to loan ratio
df["income_loan_ratio"] = (
    df["income"] / (df["loan_amount"] + 1)
)

# Log transformation of loan amount
df["log_loan_amount"] = np.log1p(
    df["loan_amount"]
)

# Debt to income ratio
df["debt_income_ratio"] = (
    df["existing_debt"] / (df["income"] + 1)
)

print("\nFEATURE ENGINEERING COMPLETED")
print(df[[
    "income_loan_ratio",
    "log_loan_amount",
    "debt_income_ratio"
]].head())

# Classification features
X = df[FEATURES].copy()

# Classification target
y = df[TARGET].copy()

# Convert target to binary if necessary
if not pd.api.types.is_numeric_dtype(y):
    target_map = {
        "yes": 1,
        "default": 1,
        "true": 1,
        "1": 1,
        "no": 0,
        "non-default": 0,
        "not default": 0,
        "false": 0,
        "0": 0
    }

    y = y.astype(str).str.strip().str.lower().map(target_map)

    if y.isna().any():
        raise ValueError(
            "Unknown values in default column. "
            "Check your target labels."
        )

y = y.astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError(
        "The default target must contain both classes 0 and 1."
    )

print("\nFEATURE COUNT:", len(FEATURES))
print("TARGET:", TARGET)

print("\nCLASS DISTRIBUTION:")
print(y.value_counts())


# ============================================================
# STEP 3: SPLIT DATA INTO TRAIN, VALIDATION AND TEST
# ============================================================

# First split: 80% development, 20% final test
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second split: 60% training, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

print("\nDATA SPLIT")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ============================================================
# STEP 4: CREATE PREPROCESSING PIPELINE
# ============================================================

def create_pipeline(classifier):

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", classifier)
    ])

    return pipeline


# ============================================================
# STEP 5: LOGISTIC REGRESSION
# ============================================================

print("\n========== LOGISTIC REGRESSION ==========")

lr_model = create_pipeline(
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_val)

lr_probability = lr_model.predict_proba(X_val)[:, 1]


# ============================================================
# STEP 6: RANDOM FOREST - GRID SEARCH CV
# ============================================================

print("\n========== RANDOM FOREST TUNING ==========")

rf_pipeline = create_pipeline(
    RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
)

rf_parameters = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_parameters,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

rf_pred = best_rf.predict(X_val)

rf_probability = best_rf.predict_proba(X_val)[:, 1]

print("\nRANDOM FOREST BEST PARAMETERS:")
print(rf_grid.best_params_)

print("RANDOM FOREST BEST CV SCORE:")
print(rf_grid.best_score_)


# ============================================================
# STEP 7: XGBOOST - RANDOMIZED SEARCH CV
# ============================================================

print("\n========== XGBOOST TUNING ==========")

positive_count = (y_train == 1).sum()
negative_count = (y_train == 0).sum()

scale_weight = negative_count / positive_count

xgb_pipeline = create_pipeline(
    XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        scale_pos_weight=scale_weight
    )
)

xgb_parameters = {
    "model__n_estimators": [100, 150, 200, 250],
    "model__max_depth": [3, 4, 5, 6],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0]
}

xgb_random = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_parameters,
    n_iter=15,
    cv=3,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_random.fit(X_train, y_train)

best_xgb = xgb_random.best_estimator_

xgb_pred = best_xgb.predict(X_val)

xgb_probability = best_xgb.predict_proba(X_val)[:, 1]

print("\nXGBOOST BEST PARAMETERS:")
print(xgb_random.best_params_)

print("XGBOOST BEST CV SCORE:")
print(xgb_random.best_score_)


# ============================================================
# STEP 8: EVALUATE ALL THREE CLASSIFIERS
# ============================================================

models = {
    "Logistic Regression": (lr_model, lr_pred, lr_probability),
    "Random Forest": (best_rf, rf_pred, rf_probability),
    "XGBoost": (best_xgb, xgb_pred, xgb_probability)
}

print("\n========== CLASSIFICATION METRICS ==========")

validation_results = {}

for name, (model, predictions, probabilities) in models.items():

    accuracy = accuracy_score(y_val, predictions)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    auc = roc_auc_score(y_val, probabilities)

    validation_results[name] = auc

    print("\nMODEL:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        predictions,
        zero_division=0
    ))


# ============================================================
# STEP 9: PLOT ROC CURVES OF THREE MODELS
# ============================================================

plt.figure(figsize=(9, 7))

for name, (model, predictions, probabilities) in models.items():

    fpr, tpr, thresholds = roc_curve(
        y_val,
        probabilities
    )

    auc = roc_auc_score(y_val, probabilities)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("task4_roc_curves.png")
plt.show()


# ============================================================
# STEP 10: DECISION THRESHOLD - AT LEAST 75% RECALL
# ============================================================

print("\n========== DECISION THRESHOLD ==========")

# Choose the best model using validation ROC-AUC
best_model_name = max(
    validation_results,
    key=validation_results.get
)

best_model = models[best_model_name][0]

best_probability = models[best_model_name][2]

print("Best Model:", best_model_name)

precision_values, recall_values, thresholds = (
    precision_recall_curve(
        y_val,
        best_probability
    )
)

# Precision and recall correspond to thresholds[:-1]
eligible = np.where(recall_values[:-1] >= 0.75)[0]

if len(eligible) > 0:

    # Select the threshold with maximum precision
    best_index = eligible[
        np.argmax(precision_values[:-1][eligible])
    ]

    best_threshold = thresholds[best_index]

    threshold_precision = precision_values[best_index]
    threshold_recall = recall_values[best_index]

    print("Selected Threshold:", round(best_threshold, 4))
    print("Precision at Threshold:", round(threshold_precision, 4))
    print("Recall at Threshold:", round(threshold_recall, 4))

else:

    best_threshold = 0.5

    print(
        "No threshold achieved 75% recall "
        "on the validation set."
    )


# ============================================================
# STEP 11: REGRESSION EVALUATION
# ============================================================

print("\n========== REGRESSION EVALUATION ==========")

# ASSUMPTION:
# loan_amount is the continuous regression target.
# Change this if your assignment uses another target column.

REGRESSION_TARGET = "loan_amount"

# Do not use features derived from loan_amount.
regression_features = [
    "age",
    "income",
    "credit_score",
    "employment_years",
    "existing_debt",
    "interest_rate",
    "loan_term"
]

missing_regression = [
    col for col in regression_features + [REGRESSION_TARGET]
    if col not in df.columns
]

if missing_regression:
    raise ValueError(
        f"Missing regression columns: {missing_regression}"
    )

reg_df = df[regression_features + [REGRESSION_TARGET]].copy()

reg_df = reg_df.replace([np.inf, -np.inf], np.nan)

reg_df[REGRESSION_TARGET] = pd.to_numeric(
    reg_df[REGRESSION_TARGET],
    errors="coerce"
)

reg_df = reg_df.dropna(subset=[REGRESSION_TARGET])

X_reg = reg_df[regression_features]
y_reg = reg_df[REGRESSION_TARGET]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )
)

# Regression model
regression_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    ))
])

regression_model.fit(X_reg_train, y_reg_train)

regression_predictions = regression_model.predict(
    X_reg_test
)

# Mean baseline
baseline_model = DummyRegressor(strategy="mean")

baseline_model.fit(X_reg_train, y_reg_train)

baseline_predictions = baseline_model.predict(
    X_reg_test
)


# Calculate regression metrics
def regression_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    return mae, rmse, r2


model_mae, model_rmse, model_r2 = regression_metrics(
    y_reg_test,
    regression_predictions
)

base_mae, base_rmse, base_r2 = regression_metrics(
    y_reg_test,
    baseline_predictions
)

print("\nRANDOM FOREST REGRESSION")
print("MAE:", round(model_mae, 4))
print("RMSE:", round(model_rmse, 4))
print("R2:", round(model_r2, 4))

print("\nMEAN BASELINE")
print("MAE:", round(base_mae, 4))
print("RMSE:", round(base_rmse, 4))
print("R2:", round(base_r2, 4))


# ============================================================
# STEP 12: FINAL TEST EVALUATION - ONLY ONCE
# ============================================================

print("\n========== FINAL TEST EVALUATION ==========")

# The test set has not been used for model selection or tuning.
# Evaluate the selected classifier only once.

final_probability = best_model.predict_proba(
    X_test
)[:, 1]

final_prediction = (
    final_probability >= best_threshold
).astype(int)

print("FINAL BEST MODEL:", best_model_name)

print(
    "Test Accuracy:",
    round(accuracy_score(y_test, final_prediction), 4)
)

print(
    "Test Precision:",
    round(precision_score(
        y_test, final_prediction, zero_division=0
    ), 4)
)

print(
    "Test Recall:",
    round(recall_score(
        y_test, final_prediction, zero_division=0
    ), 4)
)

print(
    "Test F1:",
    round(f1_score(
        y_test, final_prediction, zero_division=0
    ), 4)
)

print(
    "Test ROC-AUC:",
    round(roc_auc_score(
        y_test, final_probability
    ), 4)
)

print("\nFINAL TEST CONFUSION MATRIX:")
print(confusion_matrix(y_test, final_prediction))


# ============================================================
# STEP 13: SAVE RESULTS
# ============================================================

import joblib

joblib.dump(
    best_model,
    "task4_best_model.pkl"
)

print("\nBEST MODEL SAVED SUCCESSFULLY")

print("\n========== TASK 4 COMPLETED ==========")
