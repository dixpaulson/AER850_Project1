import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score

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

# Initialize the scaler 
scaler = StandardScaler()

# Fit the scaler on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)
                                 
# Convert the scaled features back to DataFrames
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=['X', 'Y', 'Z'])
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=['X', 'Y', 'Z'])

# Add the target variable back to the DataFrames
X_train_scaled_df['Step'] = y_train.reset_index(drop=True)
X_test_scaled_df['Step'] = y_test.reset_index(drop=True)


# Calculate the correlation matrix
corr_matrix = X_train_scaled_df.corr()
# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

# Add labels and title
plt.title('Correlation Matrix Heatmap')
plt.show()

#Support Vector Machine (SVM)
svr = SVR()
param_grid_svr = {
    'kernel': ['linear', 'rbf'],
    'C': [1],
    'gamma': ['scale', 'auto']
}
grid_search_svr = GridSearchCV(svr, param_grid_svr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_svr.fit(X_train, y_train)
best_model_svr = grid_search_svr.best_estimator_
print("Best SVM Model:", best_model_svr)

#Linear Regression
linear_reg = LinearRegression()
param_grid_lr = {}  # No hyperparameters to tune for plain linear regression, but you still apply GridSearchCV.
grid_search_lr = GridSearchCV(linear_reg, param_grid_lr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_model_lr = grid_search_lr.best_estimator_
print("Best Linear Regression Model:", best_model_lr)

# Decision Tree
decision_tree = DecisionTreeRegressor(random_state=42)
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [3, 7, 10],
    'min_samples_leaf': [1, 2, 3]
}
grid_search_dt = GridSearchCV(decision_tree, param_grid_dt, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)
best_model_dt = grid_search_dt.best_estimator_
print("Best Decision Tree Model:", best_model_dt)
