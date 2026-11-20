
# ==========================================================
# WEEK 2 - DAY 2 - TASK 2
# RANDOM FOREST CLASSIFICATION
# DATASET: WINE DATASET
# ==========================================================


# 1. IMPORT LIBRARIES

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# 2. CREATE OUTPUT FOLDER

base_dir = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(base_dir, "outputs")

os.makedirs(output_dir, exist_ok=True)


# 3. LOAD WINE DATASET

wine = load_wine()

X = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

y = wine.target

print("\n================================")
print("WINE DATASET INFORMATION")
print("================================")

print("\nFirst 5 Records:")
print(X.head())

print("\nDataset Shape:")
print(X.shape)

print("\nTarget Classes:")
print(wine.target_names)


# 4. SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# 5. TRAIN SINGLE DECISION TREE

print("\n================================")
print("SINGLE DECISION TREE")
print("================================")

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

print("\nSingle Tree Test Accuracy:")
print(round(dt_accuracy, 4))


# 6. TRAIN RANDOM FOREST WITH 200 TREES

print("\n================================")
print("RANDOM FOREST CLASSIFICATION")
print("================================")

random_forest = RandomForestClassifier(
    n_estimators=200,
    oob_score=True,
    random_state=42,
    bootstrap=True,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)


# 7. PREDICT TEST DATA

rf_predictions = random_forest.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("\nRandom Forest Test Accuracy:")
print(round(rf_accuracy, 4))


# 8. PRINT OOB SCORE

print("\nRandom Forest OOB Score:")
print(round(random_forest.oob_score_, 4))


# 9. COMPARE SINGLE TREE AND RANDOM FOREST

print("\n================================")
print("MODEL ACCURACY COMPARISON")
print("================================")

comparison = pd.DataFrame({
    "Model": [
        "Single Decision Tree",
        "Random Forest"
    ],

    "Test Accuracy": [
        dt_accuracy,
        rf_accuracy
    ]
})

print(comparison.to_string(index=False))


# 10. TRY DIFFERENT NUMBER OF TREES

print("\n================================")
print("NUMBER OF ESTIMATORS COMPARISON")
print("================================")

estimators = [10, 50, 100, 300]

accuracy_values = []

for n in estimators:

    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    accuracy_values.append(accuracy)

    print(
        f"Number of Trees: {n}, "
        f"Accuracy: {accuracy:.4f}"
    )


# 11. PLOT ACCURACY VS NUMBER OF TREES

plt.figure(figsize=(9, 5))

plt.plot(
    estimators,
    accuracy_values,
    marker="o",
    linewidth=2
)

plt.title("Random Forest Accuracy vs Number of Trees")

plt.xlabel("Number of Estimators")

plt.ylabel("Test Accuracy")

plt.xticks(estimators)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "estimators_accuracy.png"),
    dpi=300
)

plt.show()


# 12. RANDOM FOREST FEATURE IMPORTANCE

print("\n================================")
print("FEATURE IMPORTANCE")
print("================================")

feature_importance = pd.DataFrame({

    "Feature": wine.feature_names,

    "Importance": random_forest.feature_importances_

})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 5 Important Features:")

print(
    feature_importance.head(5).to_string(index=False)
)


# 13. PLOT FEATURE IMPORTANCE

plt.figure(figsize=(10, 7))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Random Forest Feature Importance")

plt.xlabel("Importance Score")

plt.ylabel("Features")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "feature_importance.png"),
    dpi=300
)

plt.show()


# 14. EXPLAIN BOOTSTRAP AGGREGATION (BAGGING)

print("\n================================")
print("BOOTSTRAP AGGREGATION (BAGGING)")
print("================================")

print("""
1. Bagging means Bootstrap Aggregation.

2. It creates multiple training datasets using
   random sampling with replacement.

3. Each dataset is used to train a separate
   decision tree.

4. The predictions from all trees are combined
   using majority voting for classification.
""")


# 15. WHY RANDOM FOREST OVERFITS LESS

print("\n================================")
print("WHY RANDOM FOREST OVERFITS LESS")
print("================================")

print("""
1. A single decision tree can memorize training data.

2. Random Forest combines multiple decision trees.

3. Each tree uses different bootstrap samples and
   random subsets of features.

4. Combining predictions reduces variance and
   generally improves generalization.

5. Therefore, Random Forest usually overfits less
   than an individual decision tree.
""")


# 16. SAVE RESULTS

comparison.to_csv(
    os.path.join(output_dir, "model_comparison.csv"),
    index=False
)

feature_importance.to_csv(
    os.path.join(output_dir, "feature_importance.csv"),
    index=False
)

estimator_results = pd.DataFrame({

    "Number of Trees": estimators,

    "Test Accuracy": accuracy_values

})

estimator_results.to_csv(
    os.path.join(output_dir, "estimators_comparison.csv"),
    index=False
)


# 17. FINAL OUTPUT

print("\n================================")
print("WEEK 2 - DAY 2 - TASK 2 COMPLETED!")
print("================================")

print("\nAll output files saved in:")
print(output_dir)
