import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Step 1: Read the image
image_path = 'your_image.png'  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found or unable to load.")
    exit()

# Convert to grayscale and preprocess
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to get binary image
_, thresh = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Find contours (components)
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Extract bounding boxes of the components
bounding_boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    bounding_boxes.append((x, y, x + w, y + h))  # (x_min, y_min, x_max, y_max)

# Step 2: Create initial grid
rows = 2  # Adjust the number of rows
cols = 2  # Adjust the number of columns

img_height, img_width = image.shape[:2]

# Initial grid lines
horizontal_lines = [int(img_height * i / rows) for i in range(1, rows)]
vertical_lines = [int(img_width * i / cols) for i in range(1, cols)]

# Step 3: Adjust grid lines to avoid intersecting components
def adjust_grid_lines(lines, bounding_boxes, is_horizontal=True):
    adjusted_lines = []
    for line in lines:
        # Check for intersection with bounding boxes
        while True:
            intersects = False
            for bbox in bounding_boxes:
                x_min, y_min, x_max, y_max = bbox
                if is_horizontal:
                    if y_min < line < y_max:
                        intersects = True
                        # Move the line up or down
                        line_candidates = [
                            y_min - 1,  # Move above the component
                            y_max + 1   # Move below the component
                        ]
                        # Choose the candidate that is closer to the initial line
                        line = min(
                            line_candidates,
                            key=lambda x: abs(x - line)
                        )
                        break
                else:
                    if x_min < line < x_max:
                        intersects = True
                        # Move the line left or right
                        line_candidates = [
                            x_min - 1,  # Move left of the component
                            x_max + 1   # Move right of the component
                        ]
                        line = min(
                            line_candidates,
                            key=lambda x: abs(x - line)
                        )
                        break
            if not intersects:
                break
        adjusted_lines.append(line)
    return adjusted_lines

# Adjust horizontal and vertical lines
adjusted_horizontal_lines = adjust_grid_lines(
    horizontal_lines, bounding_boxes, is_horizontal=True
)
adjusted_vertical_lines = adjust_grid_lines(
    vertical_lines, bounding_boxes, is_horizontal=False
)

# Step 4: Crop the image into chunks based on adjusted grid
# Define the regions based on grid lines
def get_regions(h_lines, v_lines, img_width, img_height):
    h_lines = [0] + h_lines + [img_height]
    v_lines = [0] + v_lines + [img_width]
    regions = []
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            x_min = v_lines[j]
            x_max = v_lines[j + 1]
            y_min = h_lines[i]
            y_max = h_lines[i + 1]
            regions.append((x_min, y_min, x_max, y_max))
    return regions

regions = get_regions(
    adjusted_horizontal_lines, adjusted_vertical_lines, img_width, img_height
)

# Create output directory
output_dir = 'chunks_grid_adjusted'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Crop and save the chunks
for idx, region in enumerate(regions):
    x_min, y_min, x_max, y_max = region
    chunk = image[y_min:y_max, x_min:x_max]
    chunk_path = os.path.join(output_dir, f'chunk_{idx + 1}.png')
    cv2.imwrite(chunk_path, chunk)
    print(f"Chunk {idx + 1} saved at {chunk_path}")

# Optional: Visualize the adjusted grid on the image
def visualize_grid(image, h_lines, v_lines):
    img_copy = image.copy()
    # Draw horizontal lines
    for y in h_lines:
        cv2.line(img_copy, (0, y), (img_width, y), (0, 255, 0), 1)
    # Draw vertical lines
    for x in v_lines:
        cv2.line(img_copy, (x, 0), (x, img_height), (0, 255, 0), 1)
    # Draw bounding boxes
    for bbox in bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(
            img_copy, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1
        )
    # Display the image
    cv2.imshow('Adjusted Grid', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

visualize_grid(image, adjusted_horizontal_lines, adjusted_vertical_lines)