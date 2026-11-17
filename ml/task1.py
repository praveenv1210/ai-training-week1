from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.datasets import load_iris
import pandas as pd





# Load Iris dataset
iris = load_iris()


X = iris.data
y = iris.target


df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = iris.target_names[y]


print(df.head())
print("\nDataset Shape:", df.shape)
print("\nSpecies Count:")
print(df['species'].value_counts())



# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create Logistic Regression model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=200)
)


# Train the model
model.fit(X_train, y_train)


# Predict test data
y_pred = model.predict(X_test)


# Calculate accuracy and F1 score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')


print("\nLogistic Regression Results")
print("Accuracy:", round(accuracy, 2))
print("F1 Score:", round(f1, 2))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)


print("\nConfusion Matrix:")
print(cm)



# KMeans Clustering


# Create KMeans model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


# Train using features only (no labels)
clusters = kmeans.fit_predict(X)


# Count cluster sizes
cluster_sizes = np.bincount(clusters)


print("\nKMeans Clustering Results")
print("Cluster Sizes:", cluster_sizes)



# Compare KMeans clusters with actual species


comparison = pd.crosstab(
    clusters,
    y,
    rownames=['KMeans Cluster'],
    colnames=['Actual Species']
)


comparison.columns = iris.target_names


print("\nCluster vs Actual Species:")
print(comparison)