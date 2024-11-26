import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = "/mnt/data/6C88946B-ECF3-4AD5-88E5-72A935344B3F.jpeg"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to detect black boxes (change color here by adjusting threshold values)
# For black boxes, use a low threshold value
_, threshold = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)

# Find contours of potential boxes
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter for significant box contours based on size
boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # Adjust size thresholds to filter boxes (change size criteria here)
    if w > 50 and h > 50:  # Ignore very small contours
        boxes.append((x, y, w, h))

# Sort boxes by position (top-to-bottom, left-to-right)
boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

# If no boxes are detected, treat the entire image as a single segment
if not boxes:
    boxes = [(0, 0, image.shape[1], image.shape[0])]

# Crop and save segments
segments = []
for x, y, w, h in boxes:
    cropped = image[y:y+h, x:x+w]
    segments.append(cropped)

# Display the segments
if len(segments) == 1:
    # Single segment case
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(segments[0], cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Segment 1")
else:
    # Multiple segments case
    fig, ax = plt.subplots(1, len(segments), figsize=(15, 10))
    for i, seg in enumerate(segments):
        ax[i].imshow(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))
        ax[i].axis("off")
        ax[i].set_title(f"Segment {i+1}")

plt.tight_layout()
plt.show()