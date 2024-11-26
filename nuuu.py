import cv2
import numpy as np
import os

def segment_image_watershed(image_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image '{image_path}'.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to convert the image to binary
    # Invert the image if necessary to make the foreground white
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove noise using morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Dilate to get sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Use distance transform to get sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Threshold to get sure foreground
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    # Get unknown region (area between sure background and sure foreground)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Mark unknown regions with zero
    markers[unknown == 255] = 0

    # Apply the Watershed algorithm
    markers = cv2.watershed(image, markers)
    # Mark boundaries with red color
    image[markers == -1] = [0, 0, 255]

    # Optionally, save the markers image for debugging
    markers_image = np.uint8(markers & 0xFF)
    cv2.imwrite(os.path.join(output_dir, "markers.png"), markers_image)

    # Create an output image where each segment is assigned a random color
    segments = np.zeros_like(image)
    for marker in np.unique(markers):
        if marker <= 1:
            continue
        mask = (markers == marker).astype(np.uint8) * 255
        color = np.random.randint(0, 255, size=3).tolist()
        segments[mask == 255] = color

    # Save the segmented image
    segmented_image_path = os.path.join(output_dir, "segmented_image.png")
    cv2.imwrite(segmented_image_path, segments)
    print(f"Segmented image saved to '{segmented_image_path}'.")

    # If you want to extract each segment as a separate image
    unique_markers = np.unique(markers)
    for marker in unique_markers:
        if marker <= 1:
            continue
        mask = (markers == marker).astype(np.uint8) * 255
        # Find contours to get bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Assuming one contour per segment
        x, y, w, h = cv2.boundingRect(contours[0])
        # Extract the segment from the original image
        segment = image[y:y+h, x:x+w]
        segment_mask = mask[y:y+h, x:x+w]
        segment = cv2.bitwise_and(segment, segment, mask=segment_mask)
        # Save the segment
        segment_path = os.path.join(output_dir, f"segment_{marker-1}.png")
        cv2.imwrite(segment_path, segment)
        print(f"Saved segment {marker-1} to '{segment_path}'.")

# Example usage
if __name__ == "__main__":
    image_path = "input_image.png"  # Replace with your image path
    output_dir = "segmented_output"
    segment_image_watershed(image_path, output_dir)