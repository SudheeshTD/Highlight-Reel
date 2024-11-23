import cv2
import numpy as np
import argparse
import csv
from collections import deque


def apply_effects(frame, effect_type):
    if effect_type == 'sepia':
        # Apply a sepia filter
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.transform(frame, kernel)

    elif effect_type == 'invert':
        # Invert the colors
        return cv2.bitwise_not(frame)

    elif effect_type == 'brightness':
        # Increase brightness and contrast
        return cv2.convertScaleAbs(frame, alpha=1.2, beta=30)

    elif effect_type == 'edges':
        # Convert to grayscale, blur, and apply edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    else:
        # No effect
        return frame


def process_frame(frame, apply_filters=True, crop=None, effect_type=None):
    if crop:
        h, w = frame.shape[:2]
        x0, y0, crop_w, crop_h = [int(c * 0.01 * dim) for c, dim in zip(crop, [w, h, w, h])]
        frame = frame[y0:y0+crop_h, x0:x0+crop_w]

    if apply_filters:
        frame = apply_effects(frame, effect_type)

    return frame


def apply_smoothing(predictions, window_size=5):
    smoothed = []
    window = deque(maxlen=window_size)

    for pred in predictions:
        window.append(pred)
        smoothed.append(1 if sum(window) / len(window) > 0.5 else 0)

    return smoothed


def main():
    parser = argparse.ArgumentParser(description='Process video with optional CSV frame filtering and effects')
    parser.add_argument('input_video', type=str, help='Path to the input video file')
    parser.add_argument('--csv', type=str, help='Path to CSV file for frame filtering')
    parser.add_argument('--filters', action='store_true', help='Apply video filters')
    parser.add_argument('--crop', type=int, nargs=4, metavar=('X0', 'Y0', 'W', 'H'),
                        help='Crop video as percentage: x0 y0 width height')
    parser.add_argument('--effect', type=str, choices=['sepia', 'invert', 'brightness', 'edges', 'none'],
                        default='none', help='Choose a video effect')
    args = parser.parse_args()

    # Open the input video
    input_video = cv2.VideoCapture(args.input_video)

    if not input_video.isOpened():
        print("Error opening video file")
        return

    # Get video properties
    original_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_video.get(cv2.CAP_PROP_FPS))

    # Calculate dimensions after cropping (if crop is specified)
    if args.crop:
        x0, y0, crop_w, crop_h = [int(c * 0.01 * dim) for c, dim in zip(args.crop, [original_width, original_height, original_width, original_height])]
        width, height = crop_w, crop_h
    else:
        width, height = original_width, original_height

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    # Read CSV file if provided
    frame_filter = {}
    if args.csv:
        with open(args.csv, 'r') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            frame_numbers, values = [], []
            for row in csv_reader:
                frame_numbers.append(int(row['frame']))
                values.append(int(row['value']))

        # Smooth predictions
        smoothed_values = apply_smoothing(values)
        frame_filter = dict(zip(frame_numbers, smoothed_values))

        # Find the first non-zero value frame
        frame_number = next((frame for frame, value in frame_filter.items() if value != 0), 0)

        # Set the video capture to the first non-zero value frame
        input_video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    else:
        frame_number = 0

    while True:
        ret, frame = input_video.read()

        if not ret:
            break

        # Check if we should process this frame
        if not frame_filter or frame_filter.get(frame_number, 0) == 1:
            # Process the frame
            processed_frame = process_frame(frame, args.filters, args.crop, args.effect)

            # Write the processed frame to the output video
            output_video.write(processed_frame)

            # Display the processed frame
            cv2.imshow('Processed Video', processed_frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_number += 1

    # Release resources
    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()


if __name__== "__main__":
    main()