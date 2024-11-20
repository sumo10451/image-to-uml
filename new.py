import cv2
import numpy as np

# Load the image
image = cv2.imread('input_image.jpg')

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range for orange color in HSV
lower_orange = np.array([10, 100, 100])
upper_orange = np.array([25, 255, 255])

# Create a mask for orange color
mask = cv2.inRange(hsv, lower_orange, upper_orange)

# Apply morphological operations to remove small noises
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the mask
contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

line_positions = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    # Filter based on the aspect ratio to ensure it's a line
    if w > 0.8 * image.shape[1] and h < 0.1 * image.shape[0]:
        line_positions.append(y + h // 2)  # Use the center of the line

# Sort the positions from top to bottom
line_positions = sorted(line_positions)

# Determine crop positions
height, width = image.shape[:2]
crop_positions = [0] + line_positions + [height]

# Crop the image into pieces
for i in range(len(crop_positions) - 1):
    y_start = crop_positions[i]
    y_end = crop_positions[i + 1]
    cropped_image = image[y_start:y_end, :]
    cv2.imwrite(f'cropped_{i}.jpg', cropped_image)