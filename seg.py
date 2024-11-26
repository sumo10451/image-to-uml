import cv2
import numpy as np
import os

def segment_image(image_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image '{image_path}'.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to convert the image to binary
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours and save each component
    for idx, contour in enumerate(contours):
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the component from the original image
        component = image[y:y+h, x:x+w]

        # Save the component image
        component_path = os.path.join(output_dir, f"component_{idx+1}.png")
        cv2.imwrite(component_path, component)
        print(f"Saved segmented component to '{component_path}'.")

    print(f"Total components found: {len(contours)}")

# Example usage
if __name__ == "__main__":
    image_path = "input_image.png"  # Replace with your image path
    output_dir = "segmented_components"
    segment_image(image_path, output_dir)