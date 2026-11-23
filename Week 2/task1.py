import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# Dataset path
DATA_PATH = "bank_loan_dataset.csv"
TARGET_COLUMN = "default"

# Generate sample bank loan dataset
np.random.seed(42)

n = 200

df = pd.DataFrame({
    "age": np.random.randint(21, 65, n),
    "income": np.random.randint(20000, 150000, n),
    "loan_amount": np.random.randint(50000, 1000000, n),
    "credit_score": np.random.randint(300, 850, n),
    "employment_years": np.random.randint(0, 35, n),
    "existing_debt": np.random.randint(0, 500000, n),
    "interest_rate": np.round(np.random.uniform(5, 18, n), 2),
    "loan_term": np.random.choice([12, 24, 36, 48, 60], n)
})

# Generate binary target
risk = (
    (df["credit_score"] < 600).astype(int)
    + (df["income"] < 40000).astype(int)
    + (df["existing_debt"] > 300000).astype(int)
)

df["default"] = (risk >= 2).astype(int)

# Save dataset
df.to_csv(DATA_PATH, index=False)

# Read dataset
df = pd.read_csv(DATA_PATH)

print("Dataset created successfully!")
print(df.head())
print("\nColumns:")
print(df.columns.tolist())

print("\nDataset shape:", df.shape)
print("\nTarget distribution:")
print(df[TARGET_COLUMN].value_counts())

print("Dataset shape:", df.shape)
print("Column names:", df.columns.tolist())
print("\nTarget distribution:")
print(df[TARGET_COLUMN].value_counts())

from sklearn.model_selection import train_test_split

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# First split: 60% train, 40% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# Second split: 20% validation, 20% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# Task 1: Demonstrate Overfitting

depths = [1, 3, 5, 10, None]

train_scores = []
val_scores = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    train_scores.append(train_acc)
    val_scores.append(val_acc)

    print(
        f"Depth: {depth}, "
        f"Train Accuracy: {train_acc:.2f}, "
        f"Validation Accuracy: {val_acc:.2f}"
    )

# Plot accuracy comparison

plt.figure(figsize=(9, 5))

plt.plot(depths[:4] + ["None"],
         train_scores,
         marker="o",
         label="Training Accuracy")

plt.plot(depths[:4] + ["None"],
         val_scores,
         marker="o",
         label="Validation Accuracy")

plt.xlabel("Decision Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Overfitting Analysis")
plt.legend()
plt.grid(True)

plt.show()


# Task 1: 5-Fold Cross Validation

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\n5-Fold Cross Validation Results:")
print("Individual scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())


# Task 2: Train Logistic Regression Model

model.fit(X_train, y_train)

# Validation prediction
y_val_pred = model.predict(X_val)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\nValidation Accuracy:")
print(accuracy_score(y_val, y_val_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))


# Task 3: Final Model Evaluation

y_test_pred = model.predict(X_test)

print("\nFinal Test Accuracy:")
print(accuracy_score(y_test, y_test_pred))

print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))


# Task 4: Model Comparison

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=3, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

print("\nMODEL COMPARISON")

for name, clf in models.items():
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print(f"{name}: {acc:.4f}")


# Task 7: Feature Importance Analysis

import pandas as pd
import matplotlib.pyplot as plt

# Random Forest Feature Importance
rf_model = models["Random Forest"]
rf_model.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance", ascending=False
)

print("\nFeature Importance:")
print(importance)

# Plot Feature Importance
plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Random Forest - Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



# Task 10: Model Selection using Validation Data

print("\nMODEL SELECTION - VALIDATION RESULTS")

for name, clf in models.items():
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)

    print(f"\n{name}")
    print("Validation Accuracy:",
          accuracy_score(y_val, val_pred))

    print(classification_report(y_val, val_pred))


from sklearn.metrics import f1_score
import joblib
import pandas as pd

print("\nBEST MODEL SELECTION")

best_score = -1
best_model = None
best_name = None

for name, clf in models.items():
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)

    score = f1_score(y_val, val_pred, average="macro")

    print(f"{name}: F1 Score = {score:.4f}")

    if score > best_score:
        best_score = score
        best_model = clf
        best_name = name

print("\nBest Model:", best_name)
print("Best Validation F1:", best_score)

# Train selected model on training + validation data
X_final = pd.concat([X_train, X_val])
y_final = pd.concat([y_train, y_val])

best_model.fit(X_final, y_final)

# Save model
joblib.dump(best_model, "best_bank_loan_model.pkl")

print("\nBest model saved successfully!")


# Task 12: Load Saved Model and Predict

import joblib

# Load saved model
loaded_model = joblib.load("best_bank_loan_model.pkl")

print("\nMODEL LOADED SUCCESSFULLY")

# Take one sample from test dataset
sample = X_test.iloc[[0]]

# Predict loan default
prediction = loaded_model.predict(sample)

print("\nSample Input:")
print(sample)

print("\nPrediction:")
if prediction[0] == 1:
    print("Customer is likely to DEFAULT")
else:
    print("Customer is NOT likely to DEFAULT")


# Task 13: Prediction Probability

probability = loaded_model.predict_proba(sample)

default_probability = probability[0][1]
non_default_probability = probability[0][0]

print("\nPREDICTION PROBABILITY")

print("Non-Default Probability:",
      round(non_default_probability * 100, 2), "%")

print("Default Probability:",
      round(default_probability * 100, 2), "%")

# Task 13: Final Project Summary

print("\n========== PROJECT SUMMARY ==========")

print("Project: Bank Loan Default Prediction")

print("Dataset: bank_loan_dataset.csv")

print("Best Model:", best_name)

print("Final Test Accuracy:",
      accuracy_score(y_test, y_test_pred))

print("\nProject completed successfully!")
