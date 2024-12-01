# import numpy as np
# import pandas as pd
# from itertools import product
# from dataclasses import dataclass
# from typing import List, Tuple, Dict, Optional
# import json
# import matplotlib.pyplot as plt
# from scipy.signal import medfilt
# from scipy.ndimage import gaussian_filter1d
#
# @dataclass
# class SmoothingParams:
#     window_size: int
#     filter_type: str
#     expansion_size: int
#
#     def to_dict(self) -> dict:
#         return {
#             'window_size': self.window_size,
#             'filter_type': self.filter_type,
#             'expansion_size': self.expansion_size
#         }
#
#     @classmethod
#     def from_dict(cls, d: dict) -> 'SmoothingParams':
#         return cls(
#             window_size=d['window_size'],
#             filter_type=d['filter_type'],
#             expansion_size=d['expansion_size']
#         )
#
# class PredictionSmoother:
#     def __init__(self):
#         self.default_param_grid = {
#             'window_size': [3, 5, 7, 9, 11],
#             'filter_type': ['moving_average', 'median', 'gaussian'],
#             'expansion_size': [1, 2, 3, 4, 5]
#         }
#
#     def moving_average_filter(self, predictions, window_size):
#         return np.convolve(predictions, np.ones(window_size)/window_size, mode='same')
#
#     def median_filter(self, predictions, window_size):
#         return medfilt(predictions, kernel_size=window_size)
#
#     def gaussian_filter(self, predictions, sigma):
#         return gaussian_filter1d(predictions, sigma=sigma)
#
#     def expand_frames(self, filtered_predictions, expansion_size):
#         expanded = np.zeros_like(filtered_predictions)
#         for i in range(len(filtered_predictions)):
#             start = max(0, i - expansion_size)
#             end = min(len(filtered_predictions), i + expansion_size + 1)
#             expanded[start:end] = np.maximum(expanded[start:end], filtered_predictions[i])
#         return expanded
#
#     def smooth_predictions(self, predictions: List[int], params: SmoothingParams) -> List[int]:
#         predictions = np.array(predictions)
#         if params.filter_type == 'moving_average':
#             filtered = self.moving_average_filter(predictions, params.window_size)
#         elif params.filter_type == 'median':
#             filtered = self.median_filter(predictions, params.window_size)
#         else:  # gaussian
#             filtered = self.gaussian_filter(predictions, params.window_size / 4)
#
#         expanded = self.expand_frames(filtered, params.expansion_size)
#         return (expanded > 0.5).astype(int).tolist()
#
#     def calculate_metrics(self, predictions: List[int], targets: List[int]) -> Dict[str, float]:
#         if len(predictions) != len(targets):
#             raise ValueError("Predictions and targets must have the same length")
#
#         pred_arr = np.array(predictions)
#         target_arr = np.array(targets)
#
#         accuracy = np.mean(pred_arr == target_arr)
#         pred_changes = np.sum(np.abs(np.diff(pred_arr)))
#         target_changes = np.sum(np.abs(np.diff(target_arr)))
#
#         true_pos = np.sum((pred_arr == 1) & (target_arr == 1))
#         false_pos = np.sum((pred_arr == 1) & (target_arr == 0))
#         false_neg = np.sum((pred_arr == 0) & (target_arr == 1))
#
#         precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
#         recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
#         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#
#         return {
#             'accuracy': float(accuracy),
#             'f1_score': float(f1),
#             'precision': float(precision),
#             'recall': float(recall),
#             'state_change_difference': float(abs(pred_changes - target_changes))
#         }
#
#     def optimize_parameters(self, predictions: List[int], targets: List[int], param_grid: Optional[Dict] = None, metric: str = 'f1_score') -> Tuple[SmoothingParams, Dict[str, float]]:
#         if param_grid is None:
#             param_grid = self.default_param_grid
#
#         best_params = None
#         best_metrics = None
#         best_score = -float('inf')
#
#         param_combinations = product(
#             param_grid['window_size'],
#             param_grid['filter_type'],
#             param_grid['expansion_size']
#         )
#
#         for window_size, filter_type, expansion_size in param_combinations:
#             params = SmoothingParams(window_size, filter_type, expansion_size)
#             smoothed = self.smooth_predictions(predictions, params)
#             metrics = self.calculate_metrics(smoothed, targets)
#
#             score = metrics[metric]
#             if score > best_score:
#                 best_score = score
#                 best_params = params
#                 best_metrics = metrics
#
#         return best_params, best_metrics
#
#     def save_params(self, params: SmoothingParams, filepath: str):
#         with open(filepath, 'w') as f:
#             json.dump(params.to_dict(), f, indent=2)
#
#     def load_params(self, filepath: str) -> SmoothingParams:
#         with open(filepath, 'r') as f:
#             params_dict = json.load(f)
#         return SmoothingParams.from_dict(params_dict)
#
# def plot_predictions(raw_predictions: List[int], smoothed_predictions: List[int], target_sequence: List[int], filepath: str):
#     plt.figure(figsize=(15, 6))
#     x = range(len(raw_predictions))
#     plt.step(x, raw_predictions, where='post', label='Raw Predictions', alpha=0.7)
#     plt.step(x, smoothed_predictions, where='post', label='Smoothed Predictions', alpha=0.7)
#     plt.step(x, target_sequence, where='post', label='Target Sequence', alpha=0.7)
#     plt.ylim(-0.1, 1.1)
#     plt.xlabel('Frame')
#     plt.ylabel('Prediction')
#     plt.title('Comparison of Raw and Smoothed Predictions')
#     plt.legend()
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#     plt.savefig(filepath)
#     plt.close()
#
# if __name__ == "__main__":
#     # Read CSV files
#     predictions_df = pd.read_csv('predictions_svm.csv')
#     target_df = pd.read_csv('target.csv')
#
#     # Merge dataframes on 'frame' column and sort by frame
#     merged_df = pd.merge(predictions_df, target_df, on='frame', suffixes=('_pred', '_target')).sort_values('frame')
#
#     # Convert values to integers (0 and 1)
#     raw_predictions = merged_df['value_pred'].astype(int).tolist()
#     target_sequence = merged_df['value_target'].astype(int).tolist()
#
#     # Create smoother and optimize parameters
#     smoother = PredictionSmoother()
#     best_params, best_metrics = smoother.optimize_parameters(
#         raw_predictions, target_sequence, metric='f1_score'
#     )
#
#     # Apply smoothing with best parameters
#     smoothed_predictions = smoother.smooth_predictions(raw_predictions, best_params)
#
#     # Print results
#     print("\nBest parameters found:")
#     print(json.dumps(best_params.to_dict(), indent=2))
#     print("\nMetrics with best parameters:")
#     print(json.dumps(best_metrics, indent=2))
#
#     # Create a new dataframe with smoothed predictions
#     result_df = pd.DataFrame({
#         'frame': merged_df['frame'],
#         'value': smoothed_predictions
#     })
#
#     # Write the result to a new CSV file
#     result_df.to_csv('smoothed_predictions.csv', index=False)
#     print("\nSmoothed predictions have been written to 'smoothed_predictions.csv'")
#
#     # Create and save the plot
#     plot_predictions(raw_predictions, smoothed_predictions, target_sequence, 'predictions_comparison.png')
#     print("\nPredictions comparison plot has been saved to 'predictions_comparison.png'")
#
#     # Example of saving and loading parameters
#     smoother.save_params(best_params, "best_smoothing_params.json")
#     loaded_params = smoother.load_params("best_smoothing_params.json")



import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

@dataclass
class SmoothingParams:
    window_size: int
    filter_type: str
    expansion_size: int

    def to_dict(self) -> dict:
        return {
            'window_size': self.window_size,
            'filter_type': self.filter_type,
            'expansion_size': self.expansion_size
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SmoothingParams':
        return cls(
            window_size=d['window_size'],
            filter_type=d['filter_type'],
            expansion_size=d['expansion_size']
        )

def fill_gaps(predictions, gap_size=2):
    """
    Fill gaps of 0s surrounded by 1s if the gap size is less than or equal to the threshold.
    """
    predictions = np.array(predictions)
    for i in range(1, len(predictions) - 1):
        if predictions[i] == 0:
            left = predictions[max(0, i - gap_size):i]
            right = predictions[i + 1:min(len(predictions), i + gap_size + 1)]
            if np.all(left == 1) and np.all(right == 1):
                predictions[i] = 1
    return predictions.tolist()

def expand_single_frames(predictions, expansion_size=2):
    """
    Expand isolated frames with a value of 1 to their neighbors.
    """
    predictions = np.array(predictions)
    result = predictions.copy()
    for i in range(len(predictions)):
        if predictions[i] == 1:
            start = max(0, i - expansion_size)
            end = min(len(predictions), i + expansion_size + 1)
            result[start:end] = 1
    return result.tolist()

def weighted_smoothing(predictions, weights=[0.2, 0.6, 0.2]):
    """
    Apply weighted smoothing where neighboring frames contribute to the value of the current frame.
    """
    smoothed = np.convolve(predictions, weights, mode='same')
    return (smoothed > 0.5).astype(int).tolist()

def enforce_min_persistence(predictions, min_duration=3):
    """
    Ensure that actions persist for at least a minimum duration.
    """
    predictions = np.array(predictions)
    i = 0
    while i < len(predictions):
        if predictions[i] == 1:
            start = i
            while i < len(predictions) and predictions[i] == 1:
                i += 1
            if i - start < min_duration:
                predictions[start:i] = 0  # Remove short-duration actions
        else:
            i += 1
    return predictions.tolist()

def adaptive_expansion(predictions, expansion_size=2, adaptive_factor=1):
    """
    Expand frames dynamically based on the density of actions in the neighborhood.
    """
    predictions = np.array(predictions)
    result = predictions.copy()
    for i in range(len(predictions)):
        if predictions[i] == 1:
            dynamic_expansion = expansion_size + adaptive_factor * sum(predictions[max(0, i-3):i+4])
            start = max(0, i - int(dynamic_expansion))
            end = min(len(predictions), i + int(dynamic_expansion) + 1)
            result[start:end] = 1
    return result.tolist()

class EnhancedPredictionSmoother:
    def __init__(self):
        self.default_param_grid = {
            'window_size': [3, 5, 7, 9, 11],
            'filter_type': ['moving_average', 'median', 'gaussian'],
            'expansion_size': [1, 2, 3, 4, 5]
        }

    def moving_average_filter(self, predictions, window_size):
        return np.convolve(predictions, np.ones(window_size)/window_size, mode='same')

    def median_filter(self, predictions, window_size):
        return medfilt(predictions, kernel_size=window_size)

    def gaussian_filter(self, predictions, sigma):
        return gaussian_filter1d(predictions, sigma=sigma)

    def smooth_predictions(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        predictions = np.array(predictions)
        # Initial smoothing
        if params.filter_type == 'moving_average':
            smoothed = self.moving_average_filter(predictions, params.window_size)
        elif params.filter_type == 'median':
            smoothed = self.median_filter(predictions, params.window_size)
        else:  # Gaussian
            smoothed = self.gaussian_filter(predictions, params.window_size / 4)

        # Post-processing enhancements
        smoothed = (smoothed > 0.5).astype(int)
        smoothed = fill_gaps(smoothed, gap_size=2)
        smoothed = expand_single_frames(smoothed, expansion_size=params.expansion_size)
        smoothed = weighted_smoothing(smoothed)
        smoothed = enforce_min_persistence(smoothed, min_duration=3)
        smoothed = adaptive_expansion(smoothed, expansion_size=params.expansion_size)

        return smoothed

    def save_params(self, params: SmoothingParams, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(params.to_dict(), f, indent=2)

    def load_params(self, filepath: str) -> SmoothingParams:
        with open(filepath, 'r') as f:
            params_dict = json.load(f)
        return SmoothingParams.from_dict(params_dict)

def plot_predictions(raw_predictions: List[int], smoothed_predictions: List[int], target_sequence: List[int], filepath: str):
    plt.figure(figsize=(15, 6))
    x = range(len(raw_predictions))
    plt.step(x, raw_predictions, where='post', label='Raw Predictions', alpha=0.7)
    plt.step(x, smoothed_predictions, where='post', label='Enhanced Smoothed Predictions', alpha=0.7)
    plt.step(x, target_sequence, where='post', label='Target Sequence', alpha=0.7)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Frame')
    plt.ylabel('Prediction')
    plt.title('Comparison of Raw and Enhanced Smoothed Predictions')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# Sample usage
if __name__ == "__main__":
    # Load prediction and target data
    predictions_df = pd.read_csv('predictions_svm.csv')
    target_df = pd.read_csv('target.csv')

    # Print column names for debugging
    print("Predictions DataFrame columns:", predictions_df.columns)
    print("Target DataFrame columns:", target_df.columns)

    # Check for required columns
    required_columns = {'frame', 'value'}
    if not required_columns.issubset(predictions_df.columns):
        raise KeyError(f"Missing columns in predictions_svm.csv. Expected columns: {required_columns}")
    if not required_columns.issubset(target_df.columns):
        raise KeyError(f"Missing columns in target.csv. Expected columns: {required_columns}")

    # Merge and sort data on 'frame'
    merged_df = pd.merge(predictions_df, target_df, on='frame', suffixes=('_pred', '_target')).sort_values('frame')

    # Extract predictions and targets
    raw_predictions = merged_df['value_pred'].astype(int).tolist()
    target_sequence = merged_df['value_target'].astype(int).tolist()

    # Create smoother and parameters
    smoother = EnhancedPredictionSmoother()
    params = SmoothingParams(window_size=5, filter_type='median', expansion_size=2)

    # Apply smoothing
    smoothed_predictions = smoother.smooth_predictions(raw_predictions, params)

    # Save smoothed results to CSV
    result_df = pd.DataFrame({'frame': merged_df['frame'], 'value': smoothed_predictions})
    result_df.to_csv('enhanced_smoothed_predictions.csv', index=False)

    # Plot the predictions
    plot_predictions(raw_predictions, smoothed_predictions, target_sequence, 'enhanced_predictions_comparison.png')
    print("Enhanced smoothed predictions saved to 'enhanced_smoothed_predictions.csv'")
    print("Prediction comparison plot saved as 'enhanced_predictions_comparison.png'")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# class SmoothingParams:
#     def __init__(self, window_size=5, filter_type='median', expansion_size=2, dynamic_gap=True):
#         self.window_size = window_size
#         self.filter_type = filter_type
#         self.expansion_size = expansion_size
#         self.dynamic_gap = dynamic_gap
#
#
# class PredictionSmoother:
#     def smooth_predictions(self, predictions, params):
#         predictions = np.array(predictions)
#         smoothed = predictions.copy()
#
#         if params.filter_type == 'median':
#             smoothed = self.median_filter(predictions, params.window_size)
#         elif params.filter_type == 'mean':
#             smoothed = self.mean_filter(predictions, params.window_size)
#         return smoothed.tolist()
#
#     def median_filter(self, predictions, window_size):
#         return pd.Series(predictions).rolling(window=window_size, center=True).median().fillna(0).astype(int)
#
#     def mean_filter(self, predictions, window_size):
#         return pd.Series(predictions).rolling(window=window_size, center=True).mean().round().fillna(0).astype(int)
#
#
# def fill_gaps(predictions, gap_size=2):
#     predictions = np.array(predictions)
#     for i in range(1, len(predictions) - 1):
#         if predictions[i] == 0:
#             left = predictions[max(0, i - gap_size):i]
#             right = predictions[i + 1:min(len(predictions), i + gap_size + 1)]
#             if np.all(left == 1) and np.all(right == 1):
#                 predictions[i] = 1
#     return predictions.tolist()
#
#
# def expand_single_frames(predictions, expansion_size=2):
#     predictions = np.array(predictions)
#     result = predictions.copy()
#     for i in range(len(predictions)):
#         if predictions[i] == 1:
#             start = max(0, i - expansion_size)
#             end = min(len(predictions), i + expansion_size + 1)
#             result[start:end] = 1
#     return result.tolist()
#
#
# def dynamic_gap_fill(predictions, max_gap=3):
#     """
#     Dynamically fills gaps of 0s surrounded by 1s for varying gap sizes.
#     Works directly on lists.
#     """
#     for gap in range(1, max_gap + 1):
#         predictions = fill_gaps(predictions, gap_size=gap)
#     return predictions  # Already a list, no need for `.tolist()`
#
#
# class EnhancedPredictionSmoother(PredictionSmoother):
#     def smooth_predictions(self, predictions, params: SmoothingParams):
#         # Apply initial smoothing
#         smoothed = super().smooth_predictions(predictions, params)
#
#         # Apply gap-filling dynamically
#         if params.dynamic_gap:
#             smoothed = dynamic_gap_fill(smoothed)  # No .tolist() needed
#         else:
#             smoothed = fill_gaps(smoothed, gap_size=2)
#
#         # Expand single frames
#         smoothed = expand_single_frames(smoothed, expansion_size=params.expansion_size)
#         return smoothed
#
#
# def plot_predictions(raw, smoothed, target, filename):
#     plt.figure(figsize=(12, 6))
#     plt.plot(raw, label='Raw Predictions', alpha=0.6)
#     plt.plot(smoothed, label='Enhanced Smoothed Predictions', alpha=0.8)
#     plt.plot(target, label='Target Sequence', alpha=0.8)
#     plt.legend()
#     plt.xlabel('Frame')
#     plt.ylabel('Prediction')
#     plt.title('Comparison of Raw and Enhanced Smoothed Predictions')
#     plt.savefig(filename)
#     plt.close()
#
#
# if __name__ == "__main__":
#     # Load prediction and target data
#     predictions_df = pd.read_csv('predictions_svm.csv')
#     target_df = pd.read_csv('target.csv')
#
#     # Check for required columns
#     required_columns = {'frame', 'value'}
#     if not required_columns.issubset(predictions_df.columns):
#         raise KeyError(f"Missing columns in predictions_svm.csv. Expected columns: {required_columns}")
#     if not required_columns.issubset(target_df.columns):
#         raise KeyError(f"Missing columns in target.csv. Expected columns: {required_columns}")
#
#     # Merge and sort data on 'frame'
#     merged_df = pd.merge(predictions_df, target_df, on='frame', suffixes=('_pred', '_target')).sort_values('frame')
#
#     # Extract predictions and targets
#     raw_predictions = merged_df['value_pred'].astype(int).tolist()
#     target_sequence = merged_df['value_target'].astype(int).tolist()
#
#     # Create smoother and parameters
#     smoother = EnhancedPredictionSmoother()
#     params = SmoothingParams(window_size=5, filter_type='median', expansion_size=2, dynamic_gap=True)
#
#     # Apply smoothing
#     smoothed_predictions = smoother.smooth_predictions(raw_predictions, params)
#
#     # Save smoothed results to CSV
#     result_df = pd.DataFrame({'frame': merged_df['frame'], 'value': smoothed_predictions})
#     result_df.to_csv('enhanced_smoothed_predictions.csv', index=False)
#
#     # Plot the predictions
#     plot_predictions(raw_predictions, smoothed_predictions, target_sequence, 'enhanced_predictions_comparison.png')
#     print("Enhanced smoothed predictions saved to 'enhanced_smoothed_predictions.csv'")
#     print("Prediction comparison plot saved as 'enhanced_predictions_comparison.png'")
