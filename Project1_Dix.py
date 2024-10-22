import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.ensemble import StackingClassifier
import joblib

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

# Extract coordinates and step number from the training set
x_coord = X_train['X']
y_coord = X_train['Y']
z_coord = X_train['Z']
step_values = y_train.copy()


# Plot x vs step
plt.figure(figsize=(10, 6))
plt.scatter(x_coord, step_values, c='blue', alpha=0.5)
plt.xlabel('X Coordinate')
plt.ylabel('Step Number')
plt.title('Scatter Plot of X Coordinate vs Step Number')
plt.show()

# Plot y vs step
plt.figure(figsize=(10, 6))
plt.scatter(y_coord, step_values, c='green', alpha=0.5)
plt.xlabel('Y Coordinate')
plt.ylabel('Step Number')
plt.title('Scatter Plot of Y Coordinate vs Step Number')
plt.show()

# Plot z vs step
plt.figure(figsize=(10, 6))
plt.scatter(z_coord, step_values, c='red', alpha=0.5)
plt.xlabel('Z Coordinate')
plt.ylabel('Step Number')
plt.title('Scatter Plot of Z Coordinate vs Step Number')
plt.show()



combined_train = pd.concat([X_train, y_train], axis=1)
print(combined_train)

# Calculate the correlation matrix
corr_matrix = combined_train.corr()
# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

# Add labels and title
plt.title('Correlation Matrix Heatmap')
plt.show()

# Logistic Regression Model
log_reg = LogisticRegression()
param_grid_log_reg = {'C': [10]}                                                    # HyperParameters for LogReg
grid_search_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5)               # GridSearch for LogRegression
grid_search_log_reg.fit(X_train, y_train)
best_model_log_reg = grid_search_log_reg.best_estimator_                            # Find best hyperparameter to tune based on grid search
print("Best Logistic Regression Model:", best_model_log_reg)
y_pred_log_reg = grid_search_log_reg.predict(X_test)                                # Evaluate performance of the models on the test set
acc_log_reg = accuracy_score(y_test, y_pred_log_reg)                                # Calculate accuracy
print(f'Logistic Regression Accuracy: {acc_log_reg}')                               # Print Log Reg accuracy
print("Logistic Regression Classification Report")                                  # Model Performance Analysis
print(classification_report(y_test, y_pred_log_reg))                                # Evaluate the Logistic Regression model

# Support Vector Matrix Model
svc = SVC()
param_grid_svc = {'C': [10], 'kernel': ['linear']}                           # Set Hyperparameters for SVC
grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5)                           # GridSearch for SVC
grid_search_svc.fit(X_train, y_train)
best_model_svc = grid_search_svc.best_estimator_                                    # Find best hyperparameter to tune based on grid search
print("Best SVC Model:", best_model_svc)
y_pred_svc = grid_search_svc.predict(X_test)                                        # Evaluate performance of the models on the test set
acc_svc = accuracy_score(y_test, y_pred_svc)                                        # Calculate accuracy
print(f'SVC Accuracy: {acc_svc}')                                                   # Print accuracy
print("SVC Classification Report")                                                  # Model Performance Analysis                                                  
print(classification_report(y_test, y_pred_svc))                                    # Evaluate the SVC model

#Random Forest Model
random_forest = RandomForestClassifier()                                        
param_grid_rf = {'n_estimators': [50], 'max_depth': [5]}           # Set Hyperparameters for Random Forest based on grid search
grid_search_rf = GridSearchCV(random_forest, param_grid_rf, cv=5)                   # GridSearch for Random Forest
grid_search_rf.fit(X_train, y_train)
best_model_rf = grid_search_rf.best_estimator_                                      # Find best hyperparameter to tune
print("Best Random Forest Model:", best_model_rf)
y_pred_rf = grid_search_rf.predict(X_test)                                          # Evaluate performance of the models on the test set
acc_rf = accuracy_score(y_test, y_pred_rf)                                          # Calculate accuracy
print(f'Random Forest Accuracy: {acc_rf}')                                          # Print accuracy
print("Random Forest Classification Report")                                        # Model Performance Analysis 
print(classification_report(y_test, y_pred_rf))                                     # Evaluate the Random Forest model


# RandomizedSeachCV
param_dist_rf = {
    'n_estimators': [10],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [30],
    'min_samples_split': [10],
    'min_samples_leaf': [4]
}
random_search_rf = RandomizedSearchCV(random_forest, param_dist_rf, n_iter=10, cv=5, random_state=42)
random_search_rf.fit(X_train, y_train)
best_model_random_search_rf = random_search_rf.best_estimator_                      # Find best hyperparameter to tune
print("Best Randomized Search Model:", best_model_random_search_rf)
y_pred_random_search_rf = random_search_rf.predict(X_test)                          # Evaluate performance of the models on the test set
acc_random_search_rf = accuracy_score(y_test, y_pred_random_search_rf)              # Calculate accuracy
print(f'Random Forest Randomized Search Accuracy: {acc_random_search_rf}')          # Print accuracy
print("Random Forest Randomized Search Classification Report ")                     # Model Performance Analysis
print(classification_report(y_test, y_pred_random_search_rf))                       # Evaluate the Random Forest model

#Confusion Matrix for the best model which is Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Stacked Model Performance Analysis
# Define the base models using the best estimators from grid search
base_estimators = [
    ('rf', grid_search_rf.best_estimator_),  # Best Random Forest model
    ('svc', grid_search_svc.best_estimator_)  # Best SVC model
]

# Create the Stacking Classifier with Logistic Regression as the final estimator
stacked_model = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression())

# Train the stacked model on the training data
stacked_model.fit(X_train, y_train)

# Predict on the test data using the stacked model
y_pred_stacked = stacked_model.predict(X_test)

# Evaluate the performance of the stacked model
print("Stacked Model Classification Report")
print(classification_report(y_test, y_pred_stacked))

# Generate and plot the confusion matrix for the stacked model
conf_matrix_stacked = confusion_matrix(y_test, y_pred_stacked)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_stacked, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Confusion Matrix for Stacked Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



# Model Evaluation - Save the trained stacked model
# Define the filename for the saved model
model_filename = 'stacked_model.joblib'

# Save the trained stacked model to a file
joblib.dump(stacked_model, model_filename)
print(f"Model has been saved as {model_filename}")

# Load the saved model from the file
loaded_model = joblib.load(model_filename)

# Define new coordinate data for prediction
new_coordinates = [
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
]

# Predict maintenance steps for the new coordinates using the loaded model
predictions = loaded_model.predict(new_coordinates)

# Display the predictions
print("Predicted Maintenance Steps for the new coordinates:", predictions)