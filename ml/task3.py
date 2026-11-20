
# ============================================
# WEEK 2 - DAY 2 - TASK 2
# RANDOM FOREST CLASSIFICATION
# Dataset: Wine Dataset
# ============================================

# 1. Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 2. Load Wine dataset

wine = load_wine()

X = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

y = wine.target

print("Dataset Shape:", X.shape)
print("\nFirst 5 Records:")
print(X.head())

print("\nTarget Classes:")
print(wine.target_names)


# 3. Split dataset into training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# 4. Train Single Decision Tree

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

decision_tree.fit(X_train, y_train)

dt_predictions = decision_tree.predict(X_test)

dt_accuracy = accuracy_score(
    y_test,
    dt_predictions
)

print("\nSingle Decision Tree Accuracy:",
      round(dt_accuracy, 4))


# 5. Train Random Forest (200 estimators)

random_forest = RandomForestClassifier(
    n_estimators=200,
    oob_score=True,
    random_state=42
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("\nRandom Forest Results")

print("Number of Trees:", 200)

print("OOB Score:",
      round(random_forest.oob_score_, 4))

print("Test Accuracy:",
      round(rf_accuracy, 4))


# 6. Compare Decision Tree and Random Forest

print("\nModel Comparison")

print("Decision Tree Accuracy:",
      round(dt_accuracy, 4))

print("Random Forest Accuracy:",
      round(rf_accuracy, 4))

if rf_accuracy > dt_accuracy:
    print("Random Forest performs better.")
elif rf_accuracy < dt_accuracy:
    print("Decision Tree performs better.")
else:
    print("Both models have equal accuracy.")


# 7. Test different n_estimators values

estimators = [10, 50, 100, 300]

accuracy_values = []

for n in estimators:

    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    accuracy_values.append(accuracy)

    print(
        f"n_estimators = {n}, "
        f"Accuracy = {accuracy:.4f}"
    )


# 8. Plot accuracy comparison

plt.figure(figsize=(8, 5))

plt.plot(
    estimators,
    accuracy_values,
    marker="o"
)

plt.title("Random Forest Accuracy vs Number of Trees")

plt.xlabel("Number of Estimators")

plt.ylabel("Test Accuracy")

plt.xticks(estimators)

plt.grid(True)

plt.show()


# 9. Feature Importance

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance Table")

print(feature_importance.to_string(index=False))


# 10. Plot Feature Importance

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Random Forest Feature Importance")

plt.xlabel("Importance Score")

plt.ylabel("Features")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()


# 11. Final Summary

print("\n========== TASK 2 SUMMARY ==========")

print("Dataset: Wine Dataset")

print("Total Samples:", len(X))

print("Single Decision Tree Accuracy:",
      round(dt_accuracy, 4))

print("Random Forest Accuracy:",
      round(rf_accuracy, 4))

print("Random Forest OOB Score:",
      round(random_forest.oob_score_, 4))

print("Best Number of Trees:",
      estimators[np.argmax(accuracy_values)])

print("\nTop 5 Important Features:")

print(feature_importance.head(5).to_string(index=False))

print("\nTask 2 Completed Successfully!")
