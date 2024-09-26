
import requests
import json
import os
from azure.identity import DefaultAzureCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

# Azure API endpoints and keys (Set your values)
AZURE_VISION_ENDPOINT = "https://<your-computer-vision-endpoint>.cognitiveservices.azure.com/"
AZURE_VISION_SUBSCRIPTION_KEY = os.getenv("AZURE_VISION_SUBSCRIPTION_KEY")
AZURE_OPENAI_ENDPOINT = "https://<your-openai-endpoint>.openai.azure.com/"
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
PLANTUML_SERVER = "http://www.plantuml.com/plantuml"

# Upload the image to Azure Computer Vision
def analyze_image(image_path):
    vision_url = AZURE_VISION_ENDPOINT + "vision/v3.2/analyze"
    headers = {'Ocp-Apim-Subscription-Key': AZURE_VISION_SUBSCRIPTION_KEY,
               'Content-Type': 'application/octet-stream'}
    params = {'visualFeatures': 'Objects,Description,Text'}

    # Read the image into binary
    with open(image_path, 'rb') as image:
        data = image.read()

    response = requests.post(vision_url, headers=headers, params=params, data=data)
    response.raise_for_status()
    return response.json()

# Extract text and object details from the vision response
def extract_diagram_details(vision_response):
    objects = vision_response.get('objects', [])
    text = vision_response.get('description', {}).get('captions', [{}])[0].get('text', '')

    classes = []
    relationships = []
    for obj in objects:
        class_name = obj.get('object', '')
        classes.append(class_name)

    return classes, text

# Use Azure OpenAI to generate UML description
def generate_uml_prompt(classes, description):
    openai_url = AZURE_OPENAI_ENDPOINT + "/v1/completions"
    headers = {
        'api-key': AZURE_OPENAI_KEY,
        'Content-Type': 'application/json'
    }

    # Prepare the prompt for OpenAI
    prompt = f"Convert the following components into a UML class diagram:\n"
    prompt += f"Classes: {', '.join(classes)}\n"
    prompt += f"Description: {description}\n"

    data = {
        "model": "gpt-3.5-turbo",
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.7
    }

    response = requests.post(openai_url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['choices'][0]['text']

# Use PlantUML to generate a UML diagram from text
def generate_uml_image(uml_text):
    encoded_text = requests.utils.quote(uml_text)
    uml_url = f"{PLANTUML_SERVER}/uml/{encoded_text}"
    response = requests.get(uml_url)
    response.raise_for_status()

    # Save the image
    with open("uml_diagram.png", "wb") as img_file:
        img_file.write(response.content)

# Main execution function
def convert_diagram_to_uml(image_path):
    vision_response = analyze_image(image_path)
    classes, description = extract_diagram_details(vision_response)

    # Generate UML prompt
    uml_description = generate_uml_prompt(classes, description)
    print("Generated UML Description:", uml_description)

    # (Optional) Render the UML diagram using PlantUML
    generate_uml_image(uml_description)
    print("UML Diagram saved as uml_diagram.png")

if __name__ == "__main__":
    # Path to the image file you want to analyze
    image_path = "path/to/your/diagram.png"
    convert_diagram_to_uml(image_path)
    
    
import requests
import os
import json

# Azure OpenAI API settings
AZURE_OPENAI_ENDPOINT = "https://<your-openai-endpoint>.openai.azure.com/"
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")

# Description of your diagram in textual form
diagram_description = """
Describe the diagram here. For example:
There are three classes: User, Product, and Order. User has attributes username and password and methods login() and logout(). Product has attributes name and price. Order connects User and Product indicating which user ordered which product.
"""

def call_openai_gpt_to_generate_uml(description):
    headers = {
        'Authorization': f'Bearer {AZURE_OPENAI_KEY}',
        'Content-Type': 'application/json'
    }

    data = {
        "model": "gpt-4.0-turbo",
        "prompt": f"Convert the following diagram description into a UML class diagram:\n{description}",
        "max_tokens": 200
    }

    response = requests.post(f"{AZURE_OPENAI_ENDPOINT}v1/completions", headers=headers, json=data)
    if response.status_code == 200:
        uml_description = response.json()['choices'][0]['text']
        print("Generated UML Description:", uml_description)
        return uml_description
    else:
        print("Failed to generate UML:", response.text)
        return None

if __name__ == "__main__":
    uml_description = call_openai_gpt_to_generate_uml(diagram_description)
