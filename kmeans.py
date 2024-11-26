import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read the image
image_path = 'your_image.png'  # Replace with the path to your image
image = cv2.imread(image_path)

# Check if image is loaded
if image is None:
    print("Error: Image not found or unable to load.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to get a binary image
# Adjust the threshold value if necessary
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours of the components
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract bounding boxes and centroids of the components
bounding_boxes = []
centroids = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    bounding_boxes.append([x, y, w, h])
    cx = x + w // 2
    cy = y + h // 2
    centroids.append([cx, cy])

# Convert centroids to a NumPy array
centroids = np.array(centroids)

# Decide on the number of clusters (chunks)
K = 4  # Adjust this number based on how many chunks you want

# Perform KMeans clustering on the centroids
kmeans = KMeans(n_clusters=K, random_state=42)
labels = kmeans.fit_predict(centroids)

# Group bounding boxes by their cluster labels
clusters = [[] for _ in range(K)]
for i, bbox in enumerate(bounding_boxes):
    label = labels[i]
    clusters[label].append(bbox)

# Create directory to save chunks if it doesn't exist
import os
output_dir = 'chunks'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# For each cluster, find the minimal bounding rectangle and crop the image
for i, cluster in enumerate(clusters):
    if not cluster:
        continue  # Skip empty clusters

    # Get all coordinates within the cluster
    xs = [bbox[0] for bbox in cluster]
    ys = [bbox[1] for bbox in cluster]
    ws = [bbox[2] for bbox in cluster]
    hs = [bbox[3] for bbox in cluster]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max([x + w for x, w in zip(xs, ws)])
    y_max = max([y + h for y, h in zip(ys, hs)])

    # Crop the image
    crop_img = image[y_min:y_max, x_min:x_max]

    # Save the cropped chunk
    chunk_path = os.path.join(output_dir, f'chunk_{i+1}.png')
    cv2.imwrite(chunk_path, crop_img)
    print(f"Chunk {i+1} saved at {chunk_path}")

# Optional: Visualize the clustering result
plt.figure(figsize=(8, 6))
for k in range(K):
    cluster_points = centroids[labels == k]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {k+1}')
plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
plt.legend()
plt.title('Spatial Clustering of Components')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.show()