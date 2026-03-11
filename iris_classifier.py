import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
iris = load_iris()

# Create a Pandas DataFrame for easier viewing
# iris.data contains the 4 features, iris.target contains the species labels (0, 1, 2)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print(df.head()) # Take a look at the first 5 rows

# Split the Data (Features and Target)
X = iris.data    # Features
y = iris.target  # Target labels

# Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose and Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Generate a detailed performance report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Make a Prediction on a New Flower
# Let's say we found a new Iris flower with these measurements:
# Sepal Length: 5.1, Sepal Width: 3.5, Petal Length: 1.4, Petal Width: 0.2
new_flower = [[5.1, 3.5, 1.4, 0.2]]

# Predict the species
prediction_index = model.predict(new_flower)
predicted_species = iris.target_names[prediction_index[0]]

print(f"The model predicts this flower is a: {predicted_species}")
