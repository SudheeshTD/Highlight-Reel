import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Load provided_data.csv
data = pd.read_csv('provided_data.csv', header=None, names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])

# Convert 'effort' column to numeric; non-numeric entries will be set to NaN
data['effort'] = pd.to_numeric(data['effort'], errors='coerce')

# Impute missing 'effort' values using linear interpolation
data['effort'] = data['effort'].interpolate(method='linear')

# Ensure 'frame' is integer type for merging
data['frame'] = data['frame'].astype(int)

# Load target.csv
target = pd.read_csv('target.csv')  # Assumes columns 'frame' and 'value'

# Ensure 'frame' is integer type for merging
target['frame'] = target['frame'].astype(int)

# Merge data and target on 'frame'
merged = pd.merge(data, target, on='frame', how='inner')

# Features and target
features = ['xc', 'yc', 'w', 'h', 'effort']
X = merged[features]
y = merged['value']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to create lag features for time series data
def create_lag_features(X, window_size):
    X_lagged = pd.DataFrame()
    for i in range(window_size):
        X_shifted = pd.DataFrame(X).shift(i)
        X_shifted.columns = [f"{col}_lag_{i}" for col in X_shifted.columns]
        X_lagged = pd.concat([X_lagged, X_shifted], axis=1)
    return X_lagged.dropna()

window_size = 2  # Define the window size for time series chunks
X_lagged = create_lag_features(X_scaled, window_size)
y_lagged = y.iloc[window_size - 1:]  # Adjust y to align with lagged features
frames_lagged = merged['frame'].iloc[window_size - 1:]  # Get corresponding frame numbers

# Align indices
y_lagged = y_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
frames_lagged = frames_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
X_lagged = X_lagged.reset_index(drop=True)

# Split into train and test sets (chronological split to respect time series nature)
split_index = int(len(X_lagged) * 0.7)
X_train = X_lagged.iloc[:split_index]
X_test = X_lagged.iloc[split_index:]
y_train = y_lagged.iloc[:split_index]
y_test = y_lagged.iloc[split_index:]
frames_test = frames_lagged.iloc[split_index:]  # Frames corresponding to test set

# Set up Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)

# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2, 0.5],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 1.0],
    'max_features': ['auto', 'sqrt', 'log2']
}

random_search = RandomizedSearchCV(gb_model, param_distributions=param_dist, n_iter=10, scoring='f1', cv=3, random_state=42)
random_search.fit(X_train, y_train)

# Best parameters from the random search
print("Best Parameters from RandomizedSearchCV: ", random_search.best_params_)

# Train model with best parameters found
best_gb_model = random_search.best_estimator_

# Predict on the test set
y_pred = best_gb_model.predict(X_test)

# Compute and print classification report
print(classification_report(y_test, y_pred))

# Write predictions to CSV with the same syntax as target.csv
predictions_df = pd.DataFrame({'frame': frames_test, 'value': y_pred})
predictions_df.to_csv('predictions_gb.csv', index=False)
