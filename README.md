# Highlight Reel Generation

This project automates the creation of highlight reels by combining video processing, feature engineering, and machine learning techniques. It processes video frames, applies visual effects, filters frames based on predictions, and generates a final output video.

---

##Week 1

Implemented basic frame analysis
Created visualization pipeline
Developed animation generation system
Established data processing framework

---

##Week 3

Performance Comparison
Quantitative Performance Metrics
MetricRandom ForestSVCImprovementAccuracy0.810.84+3%Weighted Avg F10.810.83+2%Class 0 F10.880.91+3%Class 1 F10.510.52+1%
Performance Analysis
The transition to SVC yielded modest but meaningful improvements across key performance indicators:

Overall Accuracy: Increased by 3 percentage points, suggesting a more reliable classification mechanism.
Weighted Average F1 Score: Improved by 2 percentage points, indicating better balanced performance across classes.
Class-Specific Performance:

Class 0 showed a notable 3% improvement
Class 1 demonstrated a slight 1% enhancement

SVC Model Configuration
Hyperparameter Selection
Our SVC implementation was carefully tuned with the following key configurations:

Regularization Strength (C): 10

This value controls the trade-off between achieving a low training error and a low testing error
A C value of 10 suggests a moderate emphasis on avoiding overfitting while maintaining model flexibility

Kernel Type: Linear

Chosen to create a straightforward decision boundary
Assumes a linearly separable relationship between features and target classes
Computationally efficient compared to non-linear kernels

Training Methodology

Cross-Validation: 3-fold

Ensures robust model evaluation by partitioning data into three segments
Helps validate model performance and generalizability
Reduces potential bias from a single train-test split

Train-Test Split: 70% training, 30% testing

Standard approach to assess model performance
Provides a reliable estimate of real-world predictive capabilities

## Features

### **1. Video Effects**

- **Available Effects**:
  - `sepia`: Adds a warm, cinematic tone.
  - `invert`: Inverts the colors of the frame.
  - `brightness`: Enhances brightness and contrast.
  - `edges`: Highlights edges using Canny edge detection.
  - `none`: Processes frames without applying any effects.

### **2. Frame Filtering**

- Filters frames based on a CSV file containing frame-value pairs:
  - Frames marked with `1` are included in the output.
  - Frames marked with `0` are skipped.
- Implements a smoothing function using a moving average to stabilize predictions.

### **3. Cropping**

- Crops video frames based on user-defined percentages (`X0`, `Y0`, `W`, `H`).
- Focuses on specific regions of interest in the video.

### **4. Edge Detection**

- Converts frames to grayscale, applies Gaussian blur, and detects edges using the Canny method.
- Combines edge-detected frames with the original for enhanced visualization.

### **5. Real-Time Frame Display**

- Displays processed frames in real-time during execution for preview purposes.

---

## Algorithms Used

### **1. Smoothing Predictions**

- Uses a moving average to reduce noise in binary predictions:
  \[
  \text{Smoothed Value} =
  \begin{cases}
  1 & \text{if } \frac{\text{Sum of Window}}{\text{Window Size}} > 0.5 \\
  0 & \text{otherwise}
  \end{cases}
  \]

### **2. Edge Detection**

- Implements the Canny edge detection algorithm:
  - Converts frames to grayscale.
  - Applies Gaussian blur to reduce noise.
  - Detects edges by identifying areas of rapid intensity change using thresholds.

### **3. Sepia Effect**

- Applies a sepia filter using a transformation kernel:
  \[
  \text{Kernel} =
  \begin{bmatrix}
  0.272 & 0.534 & 0.131 \\
  0.349 & 0.686 & 0.168 \\
  0.393 & 0.769 & 0.189
  \end{bmatrix}
  \]

### **4. Brightness Adjustment**

- Adjusts brightness and contrast using linear scaling:
  \[
  \text{Output Pixel Value} = (\text{Input Pixel Value} \times \alpha) + \beta
  \]

---

## Prerequisites

Ensure you have Python installed along with the following libraries:

````bash
pip install opencv-python-headless numpy pandas

Usage Instructions
1. Video Processor (video_processor.py)
Processes videos by applying effects, cropping, and filtering frames based on a CSV file.
Usage:
bash
python video_processor.py input_video.mp4 --csv target.csv --filters --effect sepia --crop X0 Y0 W H

Options:
Replace input_video.mp4 with your video file path.
Use --csv target.csv to specify a CSV file with frame filtering values.
Add --filters to apply visual effects.
Use --effect to specify an effect (sepia, invert, brightness, or edges).
Use --crop X0 Y0 W H to crop the video (percentages of width/height).
2. Edge Detection (opencv_intro.py)
Applies edge detection and optional cropping to a video.
Usage:
bash
python opencv_intro.py input_video.mp4 --filters --crop X0 Y0 W H

Options:
Replace input_video.mp4 with your video file path.
Add --filters to apply edge detection.
Use --crop X0 Y0 W H to crop the video (percentages of width/height).
Example Workflow
Prepare your input video (input_video.mp4) and filtering data (target.csv).
Apply effects or filtering using the following command:
bash
python video_processor.py input_video.mp4 --csv target.csv --filters --effect brightness --crop 10 10 80 80

This command will:
Apply brightness enhancement to the video.
Crop the video to focus on the central region (10% margins on all sides).
Filter frames based on predictions from target.csv.
Alternatively, use edge detection for enhanced visualization:
bash
python opencv_intro.py input_video.mp4 --filters --crop X0 Y0 W H

Tools and Libraries Used
Tool/Library	Purpose
OpenCV (cv2)	Video processing (frame extraction, effects application, video writing).
NumPy	Numerical operations for transformations and smoothing functions.
pandas	Data manipulation and CSV handling for frame filtering and predictions.


# Results: SVM vs Original Model Performance

This section outlines the performance comparison between the original model and the SVM-based model for time-series classification in the Highlight Reel Generation project. The results include precision, recall, F1-score, and accuracy metrics.

---

## **Original Model Results**

```plaintext
Week 3> python .\time_classification.py
              precision    recall  f1-score   support

           0       0.88      0.89      0.88     23037
           1       0.52      0.49      0.51      5578

    accuracy                           0.81     28615
   macro avg       0.70      0.69      0.70     28615
weighted avg       0.81      0.81      0.81     28615

Key Observations
Class 0 (Non-highlight frames):
Precision: 88%
Recall: 89%
F1-score: 88%
Class 1 (Highlight frames):
Precision: 52%
Recall: 49%
F1-score: 51%
Overall Accuracy: 81%
The original model performs well for class 0 but struggles with class 1, achieving a low recall of 49%. This indicates that many highlight frames are misclassified as non-highlight frames.
SVM-Based Model Results
text
Week 3> python .\time_classification.py

Best Parameters: {'C': 10, 'kernel': 'linear'}
              precision    recall  f1-score   support

           0       0.87      0.95      0.91     22905
           1       0.68      0.42      0.52      5711

    accuracy                           0.84     28616
   macro avg       0.77      0.69      0.71     28616
weighted avg       0.83      0.84      0.83     28616

````

\Week 4> python .\time_classification.py

              precision    recall  f1-score   support

           0       0.90      0.96      0.93     22905
           1       0.78      0.59      0.67      5711

    accuracy                           0.89     28616

macro avg 0.84 0.77 0.80 28616
weighted avg 0.88 0.89 0.88 28616

Best parameters: {'C': 10.0, 'gamma': 'scale', 'kernel': 'rbf', 'macro_f1': 0.8005280205361265, 'weighted_f1': 0.8785524332160523}
Final Weighted F1 Score on Test Data: 0.879
