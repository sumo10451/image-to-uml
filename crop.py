from PIL import Image

# Open the original image
image_path = 'topology_diagram.png'  # Replace with your image file path
im = Image.open(image_path)

# Get image dimensions
width, height = im.size

# Define grid size (number of columns and rows)
cols = 3  # Adjust based on your diagram's complexity
rows = 2  # Adjust based on your diagram's complexity

# Calculate the size of each segment
segment_width = width // cols
segment_height = height // rows

# Define overlap (in pixels)
overlap = 50  # Adjust overlap as needed

# Crop counter
crop_number = 0

# Loop over grid to create crops
for row in range(rows):
    for col in range(cols):
        left = max(col * segment_width - overlap, 0)
        upper = max(row * segment_height - overlap, 0)
        right = min((col + 1) * segment_width + overlap, width)
        lower = min((row + 1) * segment_height + overlap, height)
        
        # Define the crop area
        box = (left, upper, right, lower)
        cropped_im = im.crop(box)
        
        # Save the cropped image
        cropped_im.save(f'cropped_section_{crop_number}.png')
        crop_number += 1