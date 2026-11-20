
# ======================================================
# WEEK 2 - DAY 2 - TASK 1
# DECISION TREE CLASSIFICATION
# DATASET: WINE DATASET
# ======================================================

# 1. Import libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
    export_text
)

from sklearn.metrics import accuracy_score


# 2. Create output folder

base_dir = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(base_dir, "outputs")

os.makedirs(output_dir, exist_ok=True)


# 3. Load Wine Dataset

wine = load_wine()

X = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

y = wine.target

print("\n========== WINE DATASET ==========")

print("\nFirst 5 Records:")
print(X.head())

print("\nDataset Shape:")
print(X.shape)

print("\nTarget Classes:")
print(wine.target_names)

print("\nMissing Values:")
print(X.isnull().sum().sum())


# 4. Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# 5. Compare Gini and Entropy

print("\n========== GINI VS ENTROPY ==========")

criteria = ["gini", "entropy"]

results = []

for criterion in criteria:

    model = DecisionTreeClassifier(
        criterion=criterion,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results.append({
        "Criterion": criterion,
        "Accuracy": accuracy
    })

    print(
        criterion.upper(),
        "Accuracy:",
        round(accuracy, 4)
    )


results_df = pd.DataFrame(results)

print("\nAccuracy Comparison:")
print(results_df.to_string(index=False))


# 6. Tune max_depth and min_samples_leaf

print("\n========== HYPERPARAMETER TUNING ==========")

dt_model = DecisionTreeClassifier(
    random_state=42
)

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5, None],
    "min_samples_leaf": [1, 2, 4, 5]
}

grid_search = GridSearchCV(
    estimator=dt_model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Accuracy:")
print(round(grid_search.best_score_, 4))


# 7. Best Model Test Accuracy

best_predictions = best_model.predict(X_test)

best_accuracy = accuracy_score(
    y_test,
    best_predictions
)

print("\nBest Decision Tree Test Accuracy:")
print(round(best_accuracy, 4))


# 8. Plot Decision Tree (max_depth=3)

plt.figure(figsize=(22, 12))

plot_tree(
    best_model,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree Classification - Wine Dataset")

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "decision_tree.png"),
    dpi=300
)

plt.show()


# 9. Explain First Two Splits

print("\n========== FIRST TWO TREE SPLITS ==========")

tree_rules = export_text(
    best_model,
    feature_names=list(wine.feature_names),
    max_depth=2
)

print(tree_rules)


# 10. Top 5 Important Features

print("\n========== TOP 5 IMPORTANT FEATURES ==========")

feature_importance = pd.DataFrame({
    "Feature": wine.feature_names,
    "Importance": best_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(
    feature_importance.head(5).to_string(index=False)
)


# 11. Plot Feature Importance

top_features = feature_importance.head(5)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Decision Tree - Top 5 Feature Importance")

plt.xlabel("Importance Score")

plt.ylabel("Features")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "feature_importance.png"),
    dpi=300
)

plt.show()


# 12. Explain Gini Impurity

print("\n========== GINI IMPURITY ==========")

print("""
1. Gini impurity measures how mixed the classes
   are inside a decision tree node.

2. Gini value 0 means the node contains only one class.

3. Decision trees select splits that reduce
   Gini impurity.
""")


# 13. Save Feature Importance Table

feature_importance.to_csv(
    os.path.join(output_dir, "feature_importance.csv"),
    index=False
)


# 14. Task Completed

print("\n======================================")
print("TASK 1 COMPLETED SUCCESSFULLY!")
print("======================================")

print("\nOutput files saved in:")
print(output_dir)
