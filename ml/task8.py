
# ============================================
# WEEK 2 - DAY 3 - TASK 2
# FEATURE SCALING
# ============================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score


# --------------------------------------------
# STEP 1: CREATE AGE AND INCOME DATA
# --------------------------------------------

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [20000, 30000, 40000, 50000, 60000]
})

print("\n1. ORIGINAL DATA")
print(df)


# --------------------------------------------
# STEP 2: STANDARD SCALER
# --------------------------------------------

standard_scaler = StandardScaler()

standard_data = standard_scaler.fit_transform(df)

standard_df = pd.DataFrame(
    standard_data,
    columns=df.columns
)

print("\n2. STANDARD SCALER RESULT")
print(standard_df)

print("\nMEAN:")
print(standard_df.mean())

print("\nSTANDARD DEVIATION:")
print(standard_df.std(ddof=0))


# --------------------------------------------
# STEP 3: MIN MAX SCALER
# --------------------------------------------

minmax_scaler = MinMaxScaler()

minmax_data = minmax_scaler.fit_transform(df)

minmax_df = pd.DataFrame(
    minmax_data,
    columns=df.columns
)

print("\n3. MIN MAX SCALER RESULT")
print(minmax_df)

print("\nMINIMUM VALUES:")
print(minmax_df.min())

print("\nMAXIMUM VALUES:")
print(minmax_df.max())


# --------------------------------------------
# STEP 4: LOAD BREAST CANCER DATASET
# --------------------------------------------

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

print("\n4. BREAST CANCER DATASET")
print("Features:", X.shape)
print("Target:", y.shape)


# --------------------------------------------
# STEP 5: TRAIN TEST SPLIT
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n5. TRAIN TEST SPLIT")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------
# STEP 6: KNN WITHOUT SCALING
# --------------------------------------------

knn_without_scaling = KNeighborsClassifier(
    n_neighbors=5
)

knn_without_scaling.fit(X_train, y_train)

prediction_without_scaling = knn_without_scaling.predict(
    X_test
)

accuracy_without_scaling = accuracy_score(
    y_test,
    prediction_without_scaling
)

print("\n6. KNN WITHOUT SCALING")
print("Accuracy:", accuracy_without_scaling)


# --------------------------------------------
# STEP 7: STANDARD SCALING
# --------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# --------------------------------------------
# STEP 8: KNN WITH SCALING
# --------------------------------------------

knn_with_scaling = KNeighborsClassifier(
    n_neighbors=5
)

knn_with_scaling.fit(
    X_train_scaled,
    y_train
)

prediction_with_scaling = knn_with_scaling.predict(
    X_test_scaled
)

accuracy_with_scaling = accuracy_score(
    y_test,
    prediction_with_scaling
)

print("\n7. KNN WITH SCALING")
print("Accuracy:", accuracy_with_scaling)


# --------------------------------------------
# STEP 9: COMPARE KNN ACCURACY
# --------------------------------------------

print("\n8. KNN ACCURACY COMPARISON")

print(
    "Without scaling:",
    round(accuracy_without_scaling, 4)
)

print(
    "With scaling:",
    round(accuracy_with_scaling, 4)
)


# --------------------------------------------
# STEP 10: DECISION TREE WITHOUT SCALING
# --------------------------------------------

tree_without_scaling = DecisionTreeClassifier(
    random_state=42
)

tree_without_scaling.fit(X_train, y_train)

tree_prediction_1 = tree_without_scaling.predict(
    X_test
)

tree_accuracy_1 = accuracy_score(
    y_test,
    tree_prediction_1
)

print("\n9. DECISION TREE WITHOUT SCALING")
print("Accuracy:", tree_accuracy_1)


# --------------------------------------------
# STEP 11: DECISION TREE WITH SCALING
# --------------------------------------------

tree_with_scaling = DecisionTreeClassifier(
    random_state=42
)

tree_with_scaling.fit(
    X_train_scaled,
    y_train
)

tree_prediction_2 = tree_with_scaling.predict(
    X_test_scaled
)

tree_accuracy_2 = accuracy_score(
    y_test,
    tree_prediction_2
)

print("\n10. DECISION TREE WITH SCALING")
print("Accuracy:", tree_accuracy_2)


# --------------------------------------------
# STEP 12: DECISION TREE COMPARISON
# --------------------------------------------

print("\n11. DECISION TREE COMPARISON")

print("Without scaling:", tree_accuracy_1)

print("With scaling:", tree_accuracy_2)


# --------------------------------------------
# BONUS: ROBUST SCALER
# --------------------------------------------

outlier_data = pd.DataFrame({
    'salary': [20000, 25000, 30000, 35000, 1000000]
})

robust_scaler = RobustScaler()

robust_result = robust_scaler.fit_transform(
    outlier_data
)

robust_df = pd.DataFrame(
    robust_result,
    columns=['salary']
)

print("\n12. ORIGINAL DATA WITH OUTLIER")
print(outlier_data)

print("\n13. ROBUST SCALER RESULT")
print(robust_df)



print("\nTASK 2 COMPLETED SUCCESSFULLY!")
