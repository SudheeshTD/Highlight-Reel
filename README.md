# Highlight Reel Generation Project

## Overview

This project automates the creation of highlight reels by combining advanced video processing, feature engineering, and machine learning techniques. It leverages computer vision and statistical learning to intelligently select and enhance video highlights.

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

### 1. Video Effects

- Sepia tone
- Color inversion
- Brightness enhancement
- Edge detection

### 2. Frame Filtering

- CSV-based frame selection
- Prediction smoothing

### 3. Video Cropping

- Percentage-based region selection

### 4. Edge Detection

- Grayscale conversion
- Gaussian blur
- Canny edge method

### 5. Real-Time Preview

- Frame-by-frame processing display

## Key Algorithms

### 1. Prediction Smoothing

Reduces noise in binary predictions using a moving average.

### 2. Sepia Effect

Applies a color transformation kernel for cinematic tone.

### 3. Brightness Adjustment

Linear scaling of pixel values.

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

## Best parameters found:

{
"window_size": 5,
"filter_type": "moving_average",
"expansion_size": 5
}

## Metrics with best parameters:

{
"accuracy": 0.888139502376293,
"f1_score": 0.7217731421121252,
"precision": 0.7166033828098033,
"recall": 0.7270180353703379,
"state_change_difference": 93.0
}
