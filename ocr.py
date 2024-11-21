import os
import json
import openai
import pytesseract
from PIL import Image
from pprint import pprint

# Replace with your Azure OpenAI credentials
OPENAI_ENDPOINT = "https://<your-openai-endpoint>.openai.azure.com/"
OPENAI_KEY = "<your-openai-key>"
OPENAI_DEPLOYMENT_NAME = "<your-gpt-4-deployment-name>"

# Initialize Azure OpenAI client
openai.api_type = "azure"
openai.api_base = OPENAI_ENDPOINT
openai.api_version = "2023-05-15"
openai.api_key = OPENAI_KEY

# Define the structure to hold extracted text and classifications
data_structure = []

# Function to process the image and extract text using Tesseract OCR
def process_image(image_path):
    # Open the image using PIL
    image = Image.open(image_path)

    # Use pytesseract to extract text with bounding box data
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # Iterate through the OCR data and populate the data structure
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        if text:
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            width = ocr_data['width'][i]
            height = ocr_data['height'][i]
            confidence = ocr_data['conf'][i]

            data_structure.append({
                "text": text,
                "bounding_box": {
                    "left": x,
                    "top": y,
                    "width": width,
                    "height": height
                },
                "confidence": confidence,
                "classification": None  # Placeholder for classification
            })
    return data_structure

# Function to classify text using Azure OpenAI GPT-4
def classify_text(data_structure):
    for item in data_structure:
        prompt = f"""
        The following text was extracted from a network topology diagram:
        Text: "{item['text']}"

        Based on the text, classify it into one of the following categories:
        - Title
        - Node
        - Label
        - Connection
        - Other

        Provide only the category name.
        """

        response = openai.Completion.create(
            engine=OPENAI_DEPLOYMENT_NAME,
            prompt=prompt,
            max_tokens=5,
            n=1,
            stop=None,
            temperature=0
        )

        classification = response.choices[0].text.strip()
        item['classification'] = classification

    return data_structure

# Evaluate final data structure
def evaluate_structure(data_structure):
    evaluation = {}
    for item in data_structure:
        classification = item['classification']
        if classification not in evaluation:
            evaluation[classification] = []
        evaluation[classification].append({
            "text": item['text'],
            "bounding_box": item['bounding_box']
        })
    return evaluation

# Main execution
if __name__ == "__main__":
    image_path = input("Enter the path to the network topology image: ").strip()

    # Step 1: Process the image to extract text
    print("Processing the image and extracting text...")
    data_structure = process_image(image_path)

    # Step 2: Classify the extracted text using GPT-4
    print("Classifying the extracted text...")
    data_structure = classify_text(data_structure)

    # Step 3: Evaluate the final data structure
    print("Evaluating the data structure...")
    final_evaluation = evaluate_structure(data_structure)

    # Print the final structured data
    print("\nFinal Data Structure:")
    print(json.dumps(final_evaluation, indent=4))