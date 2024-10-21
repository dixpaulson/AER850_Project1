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

# Define the models and their hyperparameters for GridSearchCV
models = {
    'Logistic Regression': {
        'model': LogisticRegression(),
        'params': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [10, 50, 100],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    },
    'SVM': {
        'model': SVC(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    }
}

# Perform GridSearchCV for each model
best_models = {}
for model_name, model_info in models.items():
    grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_

# Perform RandomizedSearchCV for one model (e.g., Random Forest)
random_search = RandomizedSearchCV(RandomForestClassifier(), {
    'n_estimators': [10, 50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}, n_iter=10, cv=5, scoring='f1', random_state=42)
random_search.fit(X_train, y_train)
best_models['Random Forest (Randomized)'] = random_search.best_estimator_

# Evaluate the models
results = {}
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    results[model_name] = {
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'accuracy': accuracy_score(y_test, y_pred)
    }

# Print the results
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")

# Select the best model based on F1 score
best_model_name = max(results, key=lambda k: results[k]['f1_score'])
best_model = best_models[best_model_name]

# Create a confusion matrix for the best model
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f'Confusion Matrix for {best_model_name}')
plt.show()