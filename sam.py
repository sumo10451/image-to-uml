from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt

# Load the image
image_path = 'topology.png'  # Replace with your image file
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

# Load the SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Path to the model checkpoint
model_type = "vit_h"  # Use the 'vit_h' model for high quality
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

# Initialize the mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Generate masks automatically
masks = mask_generator.generate(image)

# Visualize the original image and the masks
plt.figure(figsize=(10, 10))
plt.imshow(image)

for mask in masks:
    plt.imshow(mask["segmentation"], alpha=0.5)  # Overlay each mask on the image

plt.axis("off")
plt.show()