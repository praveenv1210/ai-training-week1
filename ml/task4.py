import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("Target Shape:", y.shape)

print("Features:", data.feature_names)
print("Target Names:", data.target_names)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Training Data:", X_train.shape)
print("Validation Data:", X_val.shape)
print("Testing Data:", X_test.shape)

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=20
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("Best Iteration:", model.best_iteration)
print("Best Score:", model.best_score)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=data.target_names))


learning_rates = [0.3, 0.1, 0.03]

results = []

for lr in learning_rates:

    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=lr,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=20
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Learning Rate": lr,
        "Best Iteration": model.best_iteration,
        "Test Accuracy": accuracy
    })

result_df = pd.DataFrame(results)

print("\nLearning Rate Comparison:")
print(result_df)


importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": data.feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"][:10],
    feature_importance["Importance"][:10]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 10 XGBoost Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
