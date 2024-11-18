import cv2
import pytesseract
import openai
import json
import time
import numpy as np
from PIL import Image
import os

# -------------------------------
# Configuration and Setup
# -------------------------------

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Path to the Tesseract executable (if not in PATH)
# Uncomment and set the correct path if needed
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Example for Linux
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Example for Windows

# Image file path
IMAGE_PATH = 'topology.png'  # Replace with your image file

# Output JSON file
OUTPUT_JSON = 'topology_data.json'

# -------------------------------
# Utility Functions
# -------------------------------

def preprocess_image(image_path):
    """
    Load and preprocess the image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to get binary image
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    return image, gray, blurred, thresh

def detect_nodes(thresh_image):
    """
    Detect nodes in the thresholded image using contour detection.
    Returns a list of bounding rectangles for each node.
    """
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nodes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out small contours that may not be nodes
        if w > 30 and h > 30:
            nodes.append((x, y, w, h))
    return nodes

def extract_text_from_roi(image, roi):
    """
    Extract text from a Region of Interest (ROI) using Tesseract OCR.
    """
    x, y, w, h = roi
    roi_img = image[y:y+h, x:x+w]
    # Optional: Further preprocess ROI for better OCR accuracy
    roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    roi_blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, roi_thresh = cv2.threshold(roi_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert to PIL Image for pytesseract
    pil_img = Image.fromarray(roi_thresh)
    text = pytesseract.image_to_string(pil_img, config='--psm 6')
    return text.strip()

def detect_edges(blurred_image):
    """
    Detect edges in the blurred image using Canny Edge Detector and Hough Transform.
    Returns a list of lines detected.
    """
    edges = cv2.Canny(blurred_image, 50, 150, apertureSize=3)
    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        return lines.tolist()
    else:
        return []

def extract_arrow_direction(image, line):
    """
    Determine the direction of the arrow on a given line.
    This is a placeholder function; implementing arrow detection can be complex.
    For simplicity, we'll assume all connections are bidirectional or unidirectional based on predefined criteria.
    """
    # Placeholder: Implement arrow detection if needed
    return "Unidirectional"  # or "Bidirectional"

def extract_connection_labels(image, lines):
    """
    Extract labels from connections (edges) using OCR.
    This implementation assumes labels are near the center of the line.
    """
    connection_labels = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate the midpoint of the line
        xm, ym = (x1 + x2) // 2, (y1 + y2) // 2
        # Define a small ROI around the midpoint
        roi_size = 30
        x_start = max(xm - roi_size//2, 0)
        y_start = max(ym - roi_size//2, 0)
        roi = (x_start, y_start, roi_size, roi_size)
        label = extract_text_from_roi(image, roi)
        connection_labels.append({
            "from": None,  # To be filled later
            "to": None,    # To be filled later
            "type": "Connection",
            "label": label if label else "No Label",
            "direction": extract_arrow_direction(image, line)
        })
    return connection_labels

def map_connections_to_nodes(nodes, lines):
    """
    Map each connection (line) to source and target nodes based on proximity.
    """
    connections = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Find the closest node to the start and end points
        source = find_closest_node(nodes, (x1, y1))
        target = find_closest_node(nodes, (x2, y2))
        connections.append({
            "source": source['id'] if source else "Unknown",
            "target": target['id'] if target else "Unknown",
            "type": "Connection",
            "label": "",
            "direction": "Unidirectional"  # Default value
        })
    return connections

def find_closest_node(nodes, point):
    """
    Find the closest node to a given point.
    """
    x, y = point
    min_distance = float('inf')
    closest_node = None
    for node in nodes:
        nx, ny = node['position']
        distance = np.sqrt((nx - x)**2 + (ny - y)**2)
        if distance < min_distance:
            min_distance = distance
            closest_node = node
    return closest_node

def organize_topology_data(nodes, connections, connection_labels):
    """
    Combine nodes and connections, assigning labels to connections.
    """
    # Assign labels to connections
    for i, conn in enumerate(connections):
        if i < len(connection_labels):
            conn['label'] = connection_labels[i]['label']
            conn['direction'] = connection_labels[i]['direction']
    topology = {
        "nodes": nodes,
        "edges": connections
    }
    return topology

def save_topology_data(topology, output_path):
    """
    Save the topology data to a JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(topology, f, indent=2)
    print(f"Topology data saved to '{output_path}'.")

# -------------------------------
# GPT-4 Integration with Prompt Chaining
# -------------------------------

def call_gpt4(prompt, max_retries=5):
    """
    Call GPT-4 API with retry logic for handling rate limits.
    """
    retry_count = 0
    wait_time = 1  # Start with 1 second

    while retry_count < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a network topology analyst."},
                    {"role": "user", "content": prompt}
                ],
                timeout=30  # Optional: set a timeout for the request
            )
            return response['choices'][0]['message']['content']

        except openai.error.RateLimitError:
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retry_count += 1
            wait_time *= 2  # Exponential backoff

        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            break

    raise Exception("Max retries exceeded. Please try again later.")

def describe_nodes(topology):
    """
    Generate descriptions for each node.
    """
    nodes = topology['nodes']
    node_descriptions = "I have the following nodes in my network topology:\n\n"
    for node in nodes:
        node_descriptions += f"{node['id']}: {node['label']}\n"
    node_descriptions += "\nPlease provide a detailed description of each node, including its role and function within the network."

    return call_gpt4(node_descriptions)

def describe_connections(topology):
    """
    Generate descriptions for each connection.
    """
    edges = topology['edges']
    nodes = topology['nodes']
    connection_descriptions = "Here are the connections in my network topology:\n\n"
    for i, edge in enumerate(edges, 1):
        source_label = next((node['label'] for node in nodes if node['id'] == edge['source']), "Unknown")
        target_label = next((node['label'] for node in nodes if node['id'] == edge['target']), "Unknown")
        connection_descriptions += f"{i}. {edge['label']} from {edge['source']} ({source_label}) to {edge['target']} ({target_label})\n"
    connection_descriptions += "\nPlease provide a detailed description of each connection, explaining the data flow and interaction between the nodes."

    return call_gpt4(connection_descriptions)

def compile_report(node_desc, connection_desc):
    """
    Compile node and connection descriptions into a comprehensive report.
    """
    report_prompt = f"""
Here are the descriptions of the nodes and connections in my network topology:

**Nodes:**
{node_desc}

**Connections:**
{connection_desc}

Please compile these into a comprehensive report detailing the entire network topology, ensuring clarity and completeness.
"""

    return call_gpt4(report_prompt)

# -------------------------------
# Main Workflow
# -------------------------------

def main():
    try:
        # Step 1: Preprocess the Image
        image, gray, blurred, thresh = preprocess_image(IMAGE_PATH)
        print("Image preprocessed.")

        # Step 2: Detect Nodes
        node_rects = detect_nodes(thresh)
        print(f"Detected {len(node_rects)} nodes.")

        # Step 3: Extract Node Labels
        nodes = []
        for idx, rect in enumerate(node_rects, 1):
            x, y, w, h = rect
            label = extract_text_from_roi(image, rect)
            nodes.append({
                "id": f"Node_{idx}",
                "label": label if label else f"Unnamed_Node_{idx}",
                "position": {"x": x, "y": y}
            })
        print("Extracted node labels.")

        # Step 4: Detect Connections (Edges)
        lines = detect_edges(blurred)
        print(f"Detected {len(lines)} connections.")

        # Step 5: Map Connections to Nodes
        connections = map_connections_to_nodes(nodes, lines)
        print("Mapped connections to nodes.")

        # Step 6: Extract Connection Labels
        connection_labels = extract_connection_labels(image, lines)
        print("Extracted connection labels.")

        # Step 7: Organize Topology Data
        topology = organize_topology_data(nodes, connections, connection_labels)
        print("Organized topology data.")

        # Step 8: Save Topology Data to JSON
        save_topology_data(topology, OUTPUT_JSON)

        # Step 9: Generate Descriptions with GPT-4 using Prompt Chaining

        # Describe Nodes
        print("Generating node descriptions...")
        node_description = describe_nodes(topology)
        print("Node Descriptions:")
        print(node_description)
        time.sleep(2)  # Optional: wait between API calls

        # Describe Connections
        print("Generating connection descriptions...")
        connection_description = describe_connections(topology)
        print("Connection Descriptions:")
        print(connection_description)
        time.sleep(2)  # Optional: wait between API calls

        # Compile Comprehensive Report
        print("Compiling comprehensive report...")
        full_report = compile_report(node_description, connection_description)
        print("Comprehensive Report:")
        print(full_report)

        # Optionally, save the report to a file
        with open('topology_report.txt', 'w') as report_file:
            report_file.write(full_report)
        print("Report saved to 'topology_report.txt'.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()