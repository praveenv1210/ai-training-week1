
# ============================================
# IMPUTING AND ENCODING
# ============================================

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder
)

# --------------------------------------------
# STEP 1: CREATE DATAFRAME
# --------------------------------------------

df = pd.DataFrame({
    'age': [25, None, 40, 35, None],
    'income': [30000, 45000, None, 80000, 52000],
    'city': ['Chennai', 'Mumbai', None, 'Chennai', 'Delhi'],
    'size': ['S', 'M', 'L', 'S', 'L']
})

print("\n1. ORIGINAL DATAFRAME")
print(df)


# --------------------------------------------
# STEP 2: CHECK MISSING VALUES
# --------------------------------------------

print("\n2. MISSING VALUES")
print(df.isnull().sum())


# --------------------------------------------
# STEP 3: SIMPLE IMPUTER FOR AGE
# --------------------------------------------

age_imputer = SimpleImputer(strategy='median')

df[['age']] = age_imputer.fit_transform(df[['age']])

print("\n3. AFTER AGE IMPUTATION")
print(df)


# --------------------------------------------
# STEP 4: SIMPLE IMPUTER FOR CITY
# --------------------------------------------

city_imputer = SimpleImputer(strategy='most_frequent')

df[['city']] = city_imputer.fit_transform(df[['city']])

print("\n4. AFTER CITY IMPUTATION")
print(df)


# --------------------------------------------
# STEP 5: KNN IMPUTER
# --------------------------------------------

numeric_data = pd.DataFrame({
    'age': [25, np.nan, 40, 35, np.nan],
    'income': [30000, 45000, np.nan, 80000, 52000]
})

knn_imputer = KNNImputer(n_neighbors=2)

knn_result = knn_imputer.fit_transform(numeric_data)

knn_df = pd.DataFrame(
    knn_result,
    columns=numeric_data.columns
)

print("\n5. KNN IMPUTER RESULT")
print(knn_df)


# --------------------------------------------
# STEP 6: ONE HOT ENCODING FOR CITY
# --------------------------------------------

onehot = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

city_encoded = onehot.fit_transform(df[['city']])

city_columns = onehot.get_feature_names_out(['city'])

city_df = pd.DataFrame(
    city_encoded,
    columns=city_columns
)

print("\n6. ONE HOT ENCODED CITY")
print(city_df)


# --------------------------------------------
# STEP 7: ORDINAL ENCODING FOR SIZE
# --------------------------------------------

ordinal = OrdinalEncoder(
    categories=[['S', 'M', 'L']]
)

df[['size']] = ordinal.fit_transform(df[['size']])

print("\n7. AFTER ORDINAL ENCODING")
print(df[['size']])


# --------------------------------------------
# STEP 8: COMBINE ENCODED DATA
# --------------------------------------------

df = pd.concat(
    [df.drop(columns=['city']), city_df],
    axis=1
)

print("\n8. FINAL ENCODED DATAFRAME")
print(df)


# --------------------------------------------
# STEP 9: LABEL ENCODER
# --------------------------------------------

# The assignment does not provide a target column.
# This is an illustrative target column only.

target = pd.Series(['No', 'Yes', 'No', 'Yes', 'No'])

label_encoder = LabelEncoder()

target_encoded = label_encoder.fit_transform(target)

print("\n9. ORIGINAL TARGET")
print(target)

print("\n10. LABEL ENCODED TARGET")
print(target_encoded)

print("\nTARGET MAPPING:")
print(dict(zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
)))

print("\nTASK 1 COMPLETED SUCCESSFULLY!")
