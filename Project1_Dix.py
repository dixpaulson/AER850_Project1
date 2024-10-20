import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# Read the data from the CSV file
df = pd.read_csv('Project_1_Data.csv')

print(df)    # Print the first 5 rows of the dataframe

# Define features (X) and target (y)
X = df[['X', 'Y', 'Z']]  # Assuming 'x', 'y', 'z' are the feature columns
y = df['Step']  # Assuming 'step' is the target column

# Initialize StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# Generate the train/test indices
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Verify the distribution of classes
print("Training set class distribution:\n", y_train.value_counts(normalize=True))
print("Testing set class distribution:\n", y_test.value_counts(normalize=True))


# Extract coordinates and step number from the training set
x_train = X_train['X']
y_train = X_train['Y']
z_train = X_train['Z']
step_train = y_train


# Plot x vs step
plt.figure(figsize=(10, 6))
plt.scatter(x_train, step_train, c='blue', alpha=0.5)
plt.xlabel('X Coordinate')
plt.ylabel('Step Number')
plt.title('Scatter Plot of X Coordinate vs Step Number')
plt.show()

# Plot y vs step
plt.figure(figsize=(10, 6))
plt.scatter(y_train, step_train, c='green', alpha=0.5)
plt.xlabel('Y Coordinate')
plt.ylabel('Step Number')
plt.title('Scatter Plot of Y Coordinate vs Step Number')
plt.show()

# Plot z vs step
plt.figure(figsize=(10, 6))
plt.scatter(z_train, step_train, c='red', alpha=0.5)
plt.xlabel('Z Coordinate')
plt.ylabel('Step Number')
plt.title('Scatter Plot of Z Coordinate vs Step Number')
plt.show()

