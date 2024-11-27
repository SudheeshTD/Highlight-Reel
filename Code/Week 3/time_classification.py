import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# Load provided_data.csv
data = pd.read_csv('provided_data.csv', header=None, names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])

# Convert 'effort' column to numeric; non-numeric entries will be set to NaN
data['effort'] = pd.to_numeric(data['effort'], errors='coerce')

# Impute missing 'effort' values using linear interpolation
data['effort'] = data['effort'].interpolate(method='linear')

# Ensure 'frame' is integer type for merging
data['frame'] = data['frame'].astype(int)

# Add new features
data['aspect_ratio'] = data['w'] / data['h']  # Ratio of width to height
data['diagonal'] = np.sqrt(data['w']**2 + data['h']**2)  # Diagonal length of bounding box

# Calculate velocity and acceleration
data[['vx', 'vy']] = data[['xc', 'yc']].diff().fillna(0)  # Velocity (change in position)
data[['ax', 'ay']] = data[['vx', 'vy']].diff().fillna(0)  # Acceleration (change in velocity)

# Normalize effort relative to frame-wise mean and standard deviation
data['effort_norm'] = (data['effort'] - data['effort'].mean()) / data['effort'].std()

# Load target.csv
target = pd.read_csv('target.csv')  # Assumes columns 'frame' and 'value'

# Ensure 'frame' is integer type for merging
target['frame'] = target['frame'].astype(int)

# Merge data and target on 'frame'
merged = pd.merge(data, target, on='frame', how='inner')

# Features and target
features = ['xc', 'yc', 'w', 'h', 'effort_norm', 'aspect_ratio',
            'diagonal', 'vx', 'vy', 'ax', 'ay']
X = merged[features]
y = merged['value']

# Normalize the features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets (chronological split to respect time series nature)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Set up SVM with optimized hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],         # Regularization strength
    'kernel': ['linear'],      # Use linear kernel for efficiency on large datasets
}

grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy', n_jobs=-1)  # Parallelize with n_jobs=-1
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_svm_model = grid_search.best_estimator_

# Predict on the test set using the best SVM model
y_pred = best_svm_model.predict(X_test)

# Compute and print classification report
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred))

# Save predictions to CSV in the same format as target.csv
predictions_df = pd.DataFrame({'frame': merged.iloc[len(X_train):]['frame'],
                               'value': y_pred})
predictions_df.to_csv('predictions.csv', index=False)
