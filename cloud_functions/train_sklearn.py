"""Train a simple KNN model on the iris dataset."""

import pickle

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load data and train model
iris = load_iris()
model = KNeighborsClassifier(n_neighbors=3)
model.fit(iris.data, iris.target)

# Save model
with open("cloud_functions/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to cloud_functions/model.pkl")
print(f"Test prediction for [5.1, 3.5, 1.4, 0.2]: {model.predict([[5.1, 3.5, 1.4, 0.2]])}")
