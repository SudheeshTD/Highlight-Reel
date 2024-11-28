<!-- # Highlight Reel Generation Project

## Overview

This project automates the creation of highlight reels by combining advanced video processing, feature engineering, and machine learning techniques. It leverages computer vision and statistical learning to intelligently select and enhance video highlights.

Here is the video Link of the highlight reel: https://youtu.be/U4H8Yw5-o6s

## Project Progression

### Key Milestones

#### Week 2: Project Foundation

- Implemented basic frame analysis
- Created visualization pipeline
- Developed animation generation system
- Established data processing framework

#### Week 3: Model Performance Improvement

**Performance Comparison (Random Forest vs SVC)**

| Metric          | Random Forest | SVC  | Improvement |
| --------------- | ------------- | ---- | ----------- |
| Accuracy        | 0.81          | 0.84 | +3%         |
| Weighted Avg F1 | 0.81          | 0.83 | +2%         |
| Class 0 F1      | 0.88          | 0.91 | +3%         |
| Class 1 F1      | 0.51          | 0.52 | +1%         |

**SVC Model Configuration**:

- Regularization Strength (C): 10
- Kernel Type: Linear
- Cross-Validation: 3-fold
- Train-Test Split: 70% training, 30% testing

#### Week 4: Enhanced Performance Metrics

| Performance Metric      | Previous | New  | Improvement |
| ----------------------- | -------- | ---- | ----------- |
| **Class 0 Metrics**     |          |      |             |
| Precision               | 0.87     | 0.90 | +3.4%       |
| Recall                  | 0.95     | 0.96 | +1.1%       |
| F1-Score                | 0.91     | 0.93 | +2.2%       |
| **Class 1 Metrics**     |          |      |             |
| Precision               | 0.68     | 0.78 | +10%        |
| Recall                  | 0.42     | 0.59 | +17%        |
| F1-Score                | 0.52     | 0.67 | +15%        |
| **Overall Performance** |          |      |             |
| Accuracy                | 0.84     | 0.89 | +5%         |
| Macro Average F1        | 0.71     | 0.80 | +9%         |
| Weighted Average F1     | 0.83     | 0.88 | +5%         |

#### Week 5: Frame Processing Enhancements

- Grayscale conversion
- Gaussian blur application
- Canny edge detection
- Colored edge overlay generation
- Selective frame processing
- Percentage-based video cropping

#### Week 7: Advanced Features

**Transition Effects**:

- Fade (gamma-corrected)
- Directional wipes
- Dissolve (Perlin noise-based)
- Motion blend (optical flow)

**Prediction Processing**:

- Moving average filtering
- Median filtering
- Gaussian smoothing

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

### \*\*5. Transitions used

- Basic Transitions
  Fade
  Smoothly blends from first frame to second frame by gradually adjusting opacity
  Wipe Left
  Reveals the second frame with a moving vertical line from left to right, like opening a curtain
  Wipe Right
  Similar to wipe left but moves from right to left, replacing the first frame progressively
  Dissolve
  Randomly replaces pixels from first frame with second frame, creating a scattered effect
  Advanced Visual Transitions
  Clock Wipe
  Creates a rotating line like a clock hand that reveals the second frame in a circular motion
  Iris Wipe
  Reveals or hides the second frame using an expanding/contracting circle from the center
  Zoom
  Scales up the first frame while transitioning to the second frame, creating a zoom effect
  Pixelate
  First frame becomes increasingly pixelated until it transforms into the second frame
  Each transition can be customized using the duration_frames parameter to control its speed and smoothness.

### \*\*6. Smoothing

- Step 1: Sliding Window Majority Voting
  Uses a configurable odd-sized window (3, 5, 7, 9, or 11 frames)
  For each frame:
  Takes surrounding frames within the window
  Calculates the ratio of active states (1s) in the window
  Applies hysteresis thresholding to prevent rapid switching
  Assigns 1 if ratio exceeds threshold, 0 otherwise1
  Step 2: Minimum Duration Filtering
  Examines consecutive predictions
  Removes state changes shorter than min_duration
  If a state change duration is less than minimum:
  Reverts the entire sequence to previous state
  Maintains longer duration state changes
  Ensures stability in predictions1
  Parameters
  Key variables that control smoothing:
  window_size: Controls the sliding window length
  min_duration: Minimum frames a state must persist
  hysteresis: Threshold adjustment to prevent oscillation1
  The system automatically optimizes these parameters using grid search to find the best combination that maximizes the chosen metric (accuracy, F1 score, precision, or recall)1

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


## Prerequisites

### Required Libraries

- OpenCV (cv2)
- NumPy
- pandas

### Installation

```bash
pip install opencv-python-headless numpy pandas
```

## Usage

### Video Processor

```bash
python video_processor.py input_video.mp4 \
    --csv target.csv \
    --filters \
    --effect sepia \
    --crop X0 Y0 W H
```

### Edge Detection

```bash
python opencv_intro.py input_video.mp4 \
    --filters \
    --crop X0 Y0 W H
```

## Performance Results

### Original Model

- Class 0 Precision: 88%
- Class 0 Recall: 89%
- Class 1 Precision: 52%
- Class 1 Recall: 49%
- Overall Accuracy: 81%

### Final SVM Model

- Class 0 Precision: 90%
- Class 0 Recall: 96%
- Class 1 Precision: 78%
- Class 1 Recall: 59%
- Overall Accuracy: 89%
- Final Weighted F1 Score: 0.879

Best parameters found:
{
  "window_size": 5,
  "filter_type": "moving_average",
  "expansion_size": 5
}

Metrics with best parameters:
{
  "accuracy": 0.888139502376293,
  "f1_score": 0.7217731421121252,
  "precision": 0.7166033828098033,
  "recall": 0.7270180353703379,
  "state_change_difference": 93.0
}

```` -->

# Highlight Reel Generation Project

## Overview

This project automates the creation of highlight reels by combining advanced video processing, feature engineering, and machine learning techniques. It leverages computer vision and statistical learning to intelligently select and enhance video highlights.

[View the Highlight Reel](https://youtu.be/U4H8Yw5-o6s)

---

## Project Progression

### Key Milestones

#### Week 2: Project Foundation

- Implemented basic frame analysis.
- Created a visualization pipeline.
- Developed an animation generation system.
- Established a data processing framework.

#### Week 3: Model Performance Improvement

**Performance Comparison (Random Forest vs SVC)**

| Metric          | Random Forest | SVC  | Improvement |
| --------------- | ------------- | ---- | ----------- |
| Accuracy        | 0.81          | 0.84 | +3%         |
| Weighted Avg F1 | 0.81          | 0.83 | +2%         |
| Class 0 F1      | 0.88          | 0.91 | +3%         |
| Class 1 F1      | 0.51          | 0.52 | +1%         |

**SVC Model Configuration**:

- Regularization Strength (C): 10
- Kernel Type: Linear
- Cross-Validation: 3-fold
- Train-Test Split: 70% training, 30% testing

#### Week 4: Enhanced Performance Metrics

| Performance Metric      | Previous | New  | Improvement |
| ----------------------- | -------- | ---- | ----------- |
| **Class 0 Metrics**     |          |      |             |
| Precision               | 0.87     | 0.90 | +3.4%       |
| Recall                  | 0.95     | 0.96 | +1.1%       |
| F1-Score                | 0.91     | 0.93 | +2.2%       |
| **Class 1 Metrics**     |          |      |             |
| Precision               | 0.68     | 0.78 | +10%        |
| Recall                  | 0.42     | 0.59 | +17%        |
| F1-Score                | 0.52     | 0.67 | +15%        |
| **Overall Performance** |          |      |             |
| Accuracy                | 0.84     | 0.89 | +5%         |
| Macro Average F1        | 0.71     | 0.80 | +9%         |
| Weighted Average F1     | 0.83     | 0.88 | +5%         |

#### Week 5: Frame Processing Enhancements

- Grayscale conversion.
- Gaussian blur application.
- Canny edge detection.
- Colored edge overlay generation.
- Selective frame processing.
- Percentage-based video cropping.

#### Week 7: Advanced Features

**Transition Effects**:

- Fade (gamma-corrected)
- Directional wipes
- Dissolve (Perlin noise-based)
- Motion blend (optical flow)

**Prediction Processing**:

- Moving average filtering.
- Median filtering.
- Gaussian smoothing.

---

## Features

### 1. Video Effects

- **Available Effects**:
  - `sepia`: Adds a warm, cinematic tone.
  - `invert`: Inverts the colors of the frame.
  - `brightness`: Enhances brightness and contrast.
  - `edges`: Highlights edges using Canny edge detection.
  - `none`: Processes frames without applying any effects.

### 2. Frame Filtering

- Filters frames based on a CSV file containing frame-value pairs:
  - Frames marked with `1` are included in the output.
  - Frames marked with `0` are skipped.
- Implements a smoothing function using a moving average to stabilize predictions.

### 3. Cropping

- Crops video frames based on user-defined percentages (`X0`, `Y0`, `W`, `H`).
- Focuses on specific regions of interest in the video.

### 4. Edge Detection

- Converts frames to grayscale, applies Gaussian blur, and detects edges using the Canny method.
- Combines edge-detected frames with the original for enhanced visualization.

### 5. Real-Time Frame Display

- Displays processed frames in real-time during execution for preview purposes.

---

## Algorithms Used

### 1. Smoothing Predictions

Uses a moving average to reduce noise in binary predictions:

```math
\text{Smoothed Value} =
\begin{cases}
1 & \text{if } \frac{\text{Sum of Window}}{\text{Window Size}} > 0.5 \\
0 & \text{otherwise}
\end{cases}

```

### 2. Edge Detection

- **Canny Edge Detection**:
  - Converts frames to grayscale.
  - Applies Gaussian blur to reduce noise.
  - Identifies edges by detecting regions with rapid intensity changes.
- **Overlaying Edges**:
  - Combines edge-detected frames with the original video frames for enhanced visualization.

### 3. Cropping

- **Process**:
  - Crops each video frame based on user-defined parameters:
    - `X0`, `Y0`: Starting coordinates of the crop region.
    - `W`, `H`: Width and height of the crop region.
- **Purpose**:
  - Focuses on specific regions of interest.
  - Removes unnecessary portions of the video.

### 4. Frame Filtering

- **Input**:
  - A CSV file containing frame-value pairs:
    - `1`: Include the frame.
    - `0`: Skip the frame.
- **Filtering Mechanism**:
  - Uses smoothing techniques (e.g., moving average) to stabilize the inclusion/exclusion of frames.

### 5. Transition Effects

- **Implemented Effects**:
  - Fade (gamma correction-based)
  - Directional Wipes (left, right, up, down)
  - Dissolve (Perlin noise-based)
  - Motion Blend (optical flow-based)
- **Customization**:
  - Duration and intensity of transitions are adjustable.

### 6. Real-Time Display

- Displays processed frames during execution to provide a live preview of the applied effects and transitions.

---

## Final Results

### Model Performance

| Metric            | Previous | Improved | Difference |
| ----------------- | -------- | -------- | ---------- |
| Accuracy          | 0.81     | 0.89     | +8%        |
| Weighted Avg F1   | 0.83     | 0.88     | +5%        |
| Macro Avg F1      | 0.71     | 0.80     | +9%        |
| Class 0 Precision | 0.87     | 0.90     | +3%        |
| Class 0 Recall    | 0.95     | 0.96     | +1%        |
| Class 1 Precision | 0.68     | 0.78     | +10%       |
| Class 1 Recall    | 0.42     | 0.59     | +17%       |

---

## Code Structure

### Main Scripts

- **`video_processor.py`**:
  - Handles video effects, frame filtering, cropping, and output generation.
- **`opencv_intro.py`**:
  - Implements real-time edge detection and overlay visualization.
- **`transitions.py`**:
  - Defines custom video transition effects.

---

## How to Run the Project

1. **Setup**:

   - Install required libraries:
     ```bash
     pip install opencv-python-headless numpy pandas
     ```

2. **Run Video Processor**:

   - Process a video with effects and frame filtering:
     ```bash
     python scripts/video_processor.py data/input_video.mp4 \
         --csv data/target.csv \
         --effect sepia \
         --crop 50 50 200 200
     ```

3. **Perform Edge Detection**:

   - Generate an edge-detected video with overlays:
     ```bash
     python scripts/opencv_intro.py data/input_video.mp4 \
         --filters \
         --crop 50 50 200 200
     ```

4. **View Processed Video**:
   - Output will be saved in the `outputs/` folder:
     ```plaintext
     outputs/processed_video.mp4
     ```

---

## Challenges and Solutions

### Challenges

1. **Prediction Noise**:

   - Binary predictions had fluctuating values, causing inconsistent frame selection.

2. **Model Overfitting**:
   - High precision for Class 0 but poor recall for Class 1.

### Solutions

1. **Smoothing Techniques**:
   - Applied moving average and median filtering to stabilize predictions.
2. **Enhanced Model Tuning**:
   - Used cross-validation and adjusted regularization parameters to balance class performance.

---
