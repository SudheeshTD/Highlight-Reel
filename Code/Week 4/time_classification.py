# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, f1_score, make_scorer
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split, ParameterGrid
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
#
# # Load datasets
#
# data = pd.read_csv('provided_data.csv', header=None, names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])
# target = pd.read_csv('target.csv')  # Ensure columns 'frame' and 'value'
#
# # Data Cleaning and Feature Engineering
# data['effort'] = pd.to_numeric(data['effort'], errors='coerce')
# data['effort'] = data['effort'].interpolate(method='linear')
# data['frame'] = data['frame'].astype(int)
# target['frame'] = target['frame'].astype(int)
#
# # Merge data and target on 'frame'
# merged = pd.merge(data, target, on='frame', how='inner')
#
# # Feature Engineering
# merged['aspect_ratio'] = merged['w'] / merged['h']
# merged['size'] = merged['w'] * merged['h']
# features = ['xc', 'yc', 'w', 'h', 'effort', 'aspect_ratio', 'size']
#
# # Prepare features and target
# X = merged[features]
# y = merged['value']
#
# # Scale features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Split data into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
# X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
#     X_scaled, y, merged.index, test_size=0.3, random_state=42
# )
#
#
# # SVM Hyperparameter Tuning
# param_grid = {
#     'C': [0.1, 1, 10],  # Regularization parameter
#     'gamma': ['scale', 'auto'],  # Kernel coefficient
#     'kernel': ['rbf', 'poly', 'linear']  # Different kernel types
# }
#
# # Store all results (including F1 scores for each parameter combination)
# results = []
#
# # Iterate over parameter grid manually with tqdm progress bar
# for params in tqdm(list(ParameterGrid(param_grid)), desc="Hyperparameter Tuning Progress"):
#     # Set parameters
#     svm = SVC(**params)
#     svm.fit(X_train, y_train)
#
#     # Calculate F1 scores on test set
#     y_pred = svm.predict(X_test)
#     macro_f1 = f1_score(y_test, y_pred, average='macro')
#     weighted_f1 = f1_score(y_test, y_pred, average='weighted')
#
#     # Store results for plotting
#     results.append({'C': params['C'], 'gamma': params['gamma'], 'kernel': params['kernel'],
#                     'macro_f1': macro_f1, 'weighted_f1': weighted_f1})
#
# # Convert results to DataFrame
# results_df = pd.DataFrame(results)
#
# # Create pivot tables for macro and weighted F1 scores
# pivot_table_macro = results_df.pivot_table(values='macro_f1', index='C', columns='gamma')
# pivot_table_weighted = results_df.pivot_table(values='weighted_f1', index='C', columns='gamma')
#
# # Plot the heatmap for weighted F1 scores
# plt.figure(figsize=(12, 8))
# sns.heatmap(pivot_table_weighted, annot=True, cmap='YlGnBu', fmt='.3f')
# plt.title('SVM Hyperparameter Tuning (F1 Score)')
# plt.xlabel('Gamma')
# plt.ylabel('C')
# plt.tight_layout()
# plt.savefig('weighted_f1_heatmap.png')
# plt.show()
#
# # Optional: Plot macro F1 score as well (for comparison)
# plt.figure(figsize=(12, 8))
# sns.heatmap(pivot_table_macro, annot=True, cmap='YlGnBu', fmt='.3f')
# plt.title('SVM Hyperparameter Tuning (Macro F1 Score)')
# plt.xlabel('Gamma')
# plt.ylabel('C')
# plt.tight_layout()
# plt.savefig('macro_f1_heatmap.png')
# plt.show()
#
# # Train the best model on the entire training data
# best_params = results_df.loc[results_df['weighted_f1'].idxmax()].to_dict()
# best_svm = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
# best_svm.fit(X_train, y_train)
#
# # Evaluate on test set
# y_pred = best_svm.predict(X_test)
# print(classification_report(y_test, y_pred))
# f1 = f1_score(y_test, y_pred, average='weighted')
# print(f"Best parameters: {best_params}")
# print(f"Final Weighted F1 Score on Test Data: {f1:.3f}")
#
# predictions_df = pd.DataFrame({'frame': merged.loc[indices_test, 'frame'], 'value': y_pred})
# predictions_df.to_csv('predictions_svm.csv', index=False)

#
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# import ray
#
# # Load provided_data.csv
# data = pd.read_csv('provided_data.csv', header=None, names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])
#
# # Convert 'effort' column to numeric; non-numeric entries will be set to NaN
# data['effort'] = pd.to_numeric(data['effort'], errors='coerce')
#
# # Impute missing 'effort' values using linear interpolation
# data['effort'] = data['effort'].interpolate(method='linear')
#
# # Ensure 'frame' is integer type for merging
# data['frame'] = data['frame'].astype(int)
#
# # Load target.csv
# target = pd.read_csv('target.csv')  # Assumes columns 'frame' and 'value'
#
# # Ensure 'frame' is integer type for merging
# target['frame'] = target['frame'].astype(int)
#
# # Merge data and target on 'frame'
# merged = pd.merge(data, target, on='frame', how='inner')
#
# # Features and target
# # Spatial Features: relative_xc, relative_yc
# merged['relative_xc'] = merged['xc'] / merged['w']
# merged['relative_yc'] = merged['yc'] / merged['h']
#
# features = ['xc', 'yc', 'w', 'h', 'effort', 'relative_xc', 'relative_yc']
# X = merged[features]
# y = merged['value']
#
# # Normalize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
#
# # Function to create lag features for time series data
# def create_lag_features(X, window_size):
#     X_lagged = pd.DataFrame()
#     for i in range(window_size):
#         X_shifted = pd.DataFrame(X).shift(i)
#         X_shifted.columns = [f"{col}_lag_{i}" for col in X_shifted.columns]
#         X_lagged = pd.concat([X_lagged, X_shifted], axis=1)
#     return X_lagged.dropna()
#
#
# @ray.remote
# def create_model(X, y, window_size, C, gamma):
#     X_lagged = create_lag_features(X, window_size)
#     y_lagged = y.iloc[window_size - 1:]
#
#     # Align indices
#     y_lagged = y_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
#     X_lagged = X_lagged.reset_index(drop=True)
#
#     # Split into train and test sets
#     split_index = int(len(X_lagged) * 0.7)
#     X_train = X_lagged.iloc[:split_index]
#     X_test = X_lagged.iloc[split_index:]
#     y_train = y_lagged.iloc[:split_index]
#     y_test = y_lagged.iloc[split_index:]
#
#     # Train and evaluate the model
#     clf = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced', random_state=42)
#     clf.fit(X_train, y_train)
#     return window_size, C, gamma, clf.score(X_test, y_test)
#
#
# # Define parameter grid
# param_grid = {
#     'window_size': [1, 5, 10, 20],
#     'C': [0.1, 1, 10, 100],
#     'gamma': ['scale', 0.1, 0.01]
# }
#
# # Initialize Ray
# ray.init()
#
# # Perform grid search
# results = []
# total_iterations = len(param_grid['window_size']) * len(param_grid['C']) * len(param_grid['gamma'])
# futures = []
#
# for window_size in param_grid['window_size']:
#     for C in param_grid['C']:
#         for gamma in param_grid['gamma']:
#             futures.append(create_model.remote(X_scaled, y, window_size, C, gamma))
#
# with tqdm(total=total_iterations, desc="Parameter Search") as pbar:
#     while futures:
#         done, futures = ray.wait(futures)
#         results.extend(ray.get(done))
#         pbar.update(len(done))
#
# # Shut down Ray
# ray.shutdown()
#
# # Convert results to DataFrame
# results_df = pd.DataFrame(results, columns=['window_size', 'C', 'gamma', 'score'])
#
# # Create heatmap
# plt.figure(figsize=(12, 8))
# pivot_table = results_df.pivot(index='window_size', columns=['C', 'gamma'], values='score')
# sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.3f', cbar_kws={'label': 'Accuracy'})
# plt.title('Model Performance: Window Size vs. SVM Parameters')
# plt.xlabel('SVM Parameters (C, Gamma)')
# plt.ylabel('Window Size')
# plt.tight_layout()
# plt.savefig('svm_parameter_search_heatmap.png')
# plt.close()
#
# # Find best parameters
# best_result = results_df.loc[results_df['score'].idxmax()]
# print(f"Best parameters: Window Size = {best_result['window_size']}, "
#       f"C = {best_result['C']}, Gamma = {best_result['gamma']}")
# print(f"Best score: {best_result['score']:.3f}")
#
# # Train final model with best parameters
# best_window_size = int(best_result['window_size'])
# best_C = best_result['C']
# best_gamma = best_result['gamma']
#
# X_lagged = create_lag_features(X_scaled, best_window_size)
# y_lagged = y.iloc[best_window_size - 1:]
# frames_lagged = merged['frame'].iloc[best_window_size - 1:]
#
# # Align indices
# y_lagged = y_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
# frames_lagged = frames_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
# X_lagged = X_lagged.reset_index(drop=True)
#
# # Split into train and test sets
# split_index = int(len(X_lagged) * 0.7)
# X_train = X_lagged.iloc[:split_index]
# X_test = X_lagged.iloc[split_index:]
# y_train = y_lagged.iloc[:split_index]
# y_test = y_lagged.iloc[split_index:]
# frames_test = frames_lagged.iloc[split_index:]
#
# # Train final model
# clf = SVC(kernel='rbf', C=best_C, gamma=best_gamma, class_weight='balanced', random_state=42)
# clf.fit(X_train, y_train)
#
# # Predict on the test set
# y_pred = clf.predict(X_test)
#
# # Compute and print classification report
# print(classification_report(y_test, y_pred))
#
# # Write predictions to CSV with the same syntax as target.csv
# predictions_df = pd.DataFrame({'frame': frames_test, 'value': y_pred})
# predictions_df.to_csv('predictions_svm.csv', index=False)



















##Workingg
#
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report
# from sklearn.svm import SVC
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# import ray
#
# # Load provided_data.csv
# data = pd.read_csv('provided_data.csv', header=None, names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])
#
# # Convert 'effort' column to numeric; non-numeric entries will be set to NaN
# data['effort'] = pd.to_numeric(data['effort'], errors='coerce')
#
# # Impute missing 'effort' values using linear interpolation
# data['effort'] = data['effort'].interpolate(method='linear')
#
# # Ensure 'frame' is integer type for merging
# data['frame'] = data['frame'].astype(int)
#
# # Load target.csv
# target = pd.read_csv('target.csv')  # Assumes columns 'frame' and 'value'
#
# # Ensure 'frame' is integer type for merging
# target['frame'] = target['frame'].astype(int)
#
# # Merge data and target on 'frame'
# merged = pd.merge(data, target, on='frame', how='inner')
#
# # Features and target
# features = ['xc', 'yc', 'w', 'h', 'effort']
# X = merged[features]
# y = merged['value']
#
# # Normalize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
#
# # Function to create lag features for time series data
# def create_lag_features(X, window_size):
#     X_lagged = pd.DataFrame()
#     for i in range(window_size):
#         X_shifted = pd.DataFrame(X).shift(i)
#         X_shifted.columns = [f"{col}_lag_{i}" for col in X_shifted.columns]
#         X_lagged = pd.concat([X_lagged, X_shifted], axis=1)
#     return X_lagged.dropna()
#
#
# @ray.remote
# def create_model(X, y, window_size, C, gamma):
#     X_lagged = create_lag_features(X, window_size)
#     y_lagged = y.iloc[window_size - 1:]
#
#     # Align indices
#     y_lagged = y_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
#     X_lagged = X_lagged.reset_index(drop=True)
#
#     # Split into train and test sets
#     split_index = int(len(X_lagged) * 0.7)
#     X_train = X_lagged.iloc[:split_index]
#     X_test = X_lagged.iloc[split_index:]
#     y_train = y_lagged.iloc[:split_index]
#     y_test = y_lagged.iloc[split_index:]
#
#     # Train and evaluate the model
#     clf = SVC(C=C, gamma=gamma, random_state=42)
#     clf.fit(X_train, y_train)
#     return window_size, C, gamma, clf.score(X_test, y_test), X_test.index.tolist(), y_test.index.tolist()
#
#
# # Define parameter grid
# param_grid = {
#     'window_size': [1, 2, 5, 10, 20],
#     'C': [0.1, 1.0, 10.0],
#     'gamma': ['scale', 'auto']
# }
#
# # Initialize Ray
# ray.init(ignore_reinit_error=True)
#
# # Perform grid search
# results = []
# total_iterations = len(param_grid['window_size']) * len(param_grid['C']) * len(param_grid['gamma'])
# futures = []
#
# for window_size in param_grid['window_size']:
#     for C in param_grid['C']:
#         for gamma in param_grid['gamma']:
#             futures.append(create_model.remote(X_scaled, y, window_size, C, gamma))
#
# with tqdm(total=total_iterations, desc="Running Grid Search") as pbar:
#     while futures:
#         done, futures = ray.wait(futures)
#         results.extend(ray.get(done))
#         pbar.update(len(done))
#
# # Shut down Ray
# ray.shutdown()
#
# # Convert results to DataFrame
# results_df = pd.DataFrame(results, columns=['window_size', 'C', 'gamma', 'score', 'test_index', 'y_test_index'])
#
# # Find best parameters
# best_result = results_df.loc[results_df['score'].idxmax()]
# print(
#     f"Best parameters: Window Size = {best_result['window_size']}, C = {best_result['C']}, Gamma = {best_result['gamma']}")
# print(f"Best accuracy: {best_result['score']:.3f}")
#
# # Train the final model using the best parameters
# best_window_size = int(best_result['window_size'])
# best_C = float(best_result['C'])
# best_gamma = best_result['gamma']
#
# # Apply lag features
# X_lagged = create_lag_features(X_scaled, best_window_size)
# y_lagged = y.iloc[best_window_size - 1:]
#
# # Align indices to ensure the same length
# X_lagged = X_lagged.reset_index(drop=True)
# y_lagged = y_lagged.reset_index(drop=True)
#
# # Split data into training and testing sets
# split_index = int(len(X_lagged) * 0.7)
# X_train, X_test = X_lagged.iloc[:split_index], X_lagged.iloc[split_index:]
# y_train, y_test = y_lagged.iloc[:split_index], y_lagged.iloc[split_index:]
#
# # Train final SVM model
# clf = SVC(C=best_C, gamma=best_gamma, random_state=42)
# clf.fit(X_train, y_train)
#
# # Predict on the test set
# y_pred = clf.predict(X_test)
#
# # Print classification report
# from sklearn.metrics import classification_report
#
# print(classification_report(y_test, y_pred))
#
# # Use the indices from the best result to fetch corresponding frame values
# test_frames = merged['frame'].iloc[best_result['y_test_index']].reset_index(drop=True)
#
# # Save predictions to CSV (Fix the mismatch issue here)
# predictions_df = pd.DataFrame({
#     'frame': test_frames,  # Correct alignment
#     'value': y_pred
# })
#
# # Save the predictions to CSV
# predictions_df.to_csv('predictions_svm.csv', index=False)












import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ray

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

@ray.remote
def create_model(X, y, window_size, C, gamma):
    X_lagged = create_lag_features(X, window_size)
    y_lagged = y.iloc[window_size - 1:]

    # Align indices
    y_lagged = y_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
    X_lagged = X_lagged.reset_index(drop=True)

    # Split into train and test sets
    split_index = int(len(X_lagged) * 0.7)
    X_train = X_lagged.iloc[:split_index]
    X_test = X_lagged.iloc[split_index:]
    y_train = y_lagged.iloc[:split_index]
    y_test = y_lagged.iloc[split_index:]

    # Train and evaluate the model
    clf = SVC(C=C, gamma=gamma, random_state=42)
    clf.fit(X_train, y_train)
    return window_size, C, gamma, clf.score(X_test, y_test), X_test.index.tolist(), y_test.index.tolist()

# Define parameter grid
param_grid = {
    'window_size': [1, 2, 5, 10, 20],
    'C': [0.1, 1.0, 10.0],
    'gamma': ['scale', 'auto']
}

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Perform grid search
results = []
total_iterations = len(param_grid['window_size']) * len(param_grid['C']) * len(param_grid['gamma'])
futures = []

for window_size in param_grid['window_size']:
    for C in param_grid['C']:
        for gamma in param_grid['gamma']:
            futures.append(create_model.remote(X_scaled, y, window_size, C, gamma))

with tqdm(total=total_iterations, desc="Running Grid Search") as pbar:
    while futures:
        done, futures = ray.wait(futures)
        results.extend(ray.get(done))
        pbar.update(len(done))

# Shut down Ray
ray.shutdown()

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=['window_size', 'C', 'gamma', 'score', 'test_index', 'y_test_index'])

# Find best parameters
best_result = results_df.loc[results_df['score'].idxmax()]
print(
    f"Best parameters: Window Size = {best_result['window_size']}, C = {best_result['C']}, Gamma = {best_result['gamma']}")
print(f"Best accuracy: {best_result['score']:.3f}")

# Train the final model using the best parameters
best_window_size = int(best_result['window_size'])
best_C = float(best_result['C'])
best_gamma = best_result['gamma']

# Apply lag features
X_lagged = create_lag_features(X_scaled, best_window_size)
y_lagged = y.iloc[best_window_size - 1:]

# Align indices to ensure the same length
X_lagged = X_lagged.reset_index(drop=True)
y_lagged = y_lagged.reset_index(drop=True)

# Split data into training and testing sets
split_index = int(len(X_lagged) * 0.7)
X_train, X_test = X_lagged.iloc[:split_index], X_lagged.iloc[split_index:]
y_train, y_test = y_lagged.iloc[:split_index], y_lagged.iloc[split_index:]

# Train final SVM model
clf = SVC(C=best_C, gamma=best_gamma, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Use the indices from the best result to fetch corresponding frame values
test_frames = merged['frame'].iloc[best_result['y_test_index']].reset_index(drop=True)

# Save predictions to CSV
predictions_df = pd.DataFrame({
    'frame': test_frames,  # Correct alignment
    'value': y_pred
})


if predictions_df.empty:
    print("The DataFrame is empty. No CSV file will be created.")
else:
    predictions_df.to_csv('predictions_svm.csv', index=False)
    print("File saved successfully.")


# Create pivot tables for macro and weighted F1 scores
from sklearn.metrics import precision_recall_fscore_support

# Get the precision, recall, and f1 scores for each class
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

# Create a DataFrame with the results
f1_scores = pd.DataFrame({
    'class': np.unique(y_test),
    'macro_f1': f1,
    'weighted_f1': f1_score(y_test, y_pred, average='weighted')
})

# Pivot table for weighted F1 scores
f1_pivot = f1_scores.pivot(index='class', columns='macro_f1', values='weighted_f1')
print(f1_pivot)

# Plot the heatmap for weighted F1 scores
plt.figure(figsize=(10, 6))
sns.heatmap(f1_pivot, annot=True, cmap='Blues')
plt.title('Heatmap of Weighted F1 Scores')
plt.show()

# Optional: Plot macro F1 score as well (for comparison)
plt.figure(figsize=(10, 6))
sns.heatmap(f1_scores[['class', 'macro_f1']].set_index('class').T, annot=True, cmap='Blues')
plt.title('Heatmap of Macro F1 Scores')
plt.show()
