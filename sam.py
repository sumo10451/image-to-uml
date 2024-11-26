# Import necessary libraries
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# Function to display masks on the image
def show_masks(image, masks, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask_data in masks:
        mask = mask_data['segmentation']
        color = np.random.random(3)  # Random color for each mask
        plt.imshow(np.dstack((mask, mask, mask)) * color.reshape(1, 1, -1), alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to display a single mask
def show_mask(mask, image, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    color = np.array([30/255, 144/255, 255/255, 0.6])  # Default color
    mask_image = np.zeros((mask.shape[0], mask.shape[1], 4))
    mask_image[:, :, :3] = color[:3]
    mask_image[:, :, 3] = mask * color[3]
    plt.imshow(mask_image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Specify the model type and path to the checkpoint
model_type = "vit_h"  # Choose from 'vit_h', 'vit_l', 'vit_b'
sam_checkpoint = "path/to/sam_vit_h_4b8939.pth"  # Update with your checkpoint path

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Load your topology image
image_path = 'path/to/your/topology_image.png'  # Update with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# =============================
# Automatic Mask Generation
# =============================

# Create a mask generator and generate masks
mask_generator = SamAutomaticMaskGenerator(sam)
masks_auto = mask_generator.generate(image)

# Visualize the automatic mask generation results
show_masks(image, masks_auto, title="Automatic Mask Generation Results")

# =============================
# Interactive Segmentation with Prompts
# =============================

# Initialize the predictor
predictor = SamPredictor(sam)
predictor.set_image(image)

# Define interactive points and labels
# Replace these with your own points and labels
input_points = np.array([
    [100, 200],  # Point 1 coordinates (x, y)
    [150, 250],  # Point 2 coordinates (x, y)
    [200, 300],  # Point 3 coordinates (x, y)
])
input_labels = np.array([1, 1, 1])  # 1 for foreground

# Perform prediction
masks_interactive, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False,  # Set to False to get a single mask
)

# Visualize the interactive segmentation results
for i, mask in enumerate(masks_interactive):
    show_mask(mask, image, title=f"Interactive Segmentation Result {i+1}")

# Alternatively, overlay all masks
masks_data = [{'segmentation': mask} for mask in masks_interactive]
show_masks(image, masks_data, title="Interactive Segmentation Results")