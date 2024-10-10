import openpyxl
import requests
import json

# Load the Excel file
def load_excel(file_path):
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    data = []
    for row in sheet.iter_rows(min_row=2, values_only=True):  # Skipping the header row
        data.append({
            'name': row[0],
            'description': row[1],
            'versionSpec': row[2],
            'publisherName': row[3],
        })
    return data

# Get the tech publisher ID based on the publisher name
def get_tech_publisher_id(publisher_name, api_url, headers):
    tech_publisher_id = None
    cursor = None
    while True:
        query = {
            "query": """
            query ListTechPublishers($first: Int!, $cursor: String) {
                techPublishers(
                    after: $cursor
                    first: $first
                    order: { name: ASC }
                ) {
                    edges {
                        node {
                            id
                            name
                            deleted
                        }
                        cursor
                    }
                    pageInfo {
                        hasNextPage
                        endCursor
                    }
                }
            }
            """,
            "variables": {
                "first": 100,
                "cursor": cursor
            }
        }
        response = requests.post(api_url, headers=headers, json=query)
        response_data = response.json()

        for edge in response_data['data']['techPublishers']['edges']:
            node = edge['node']
            if node['name'] == publisher_name and not node['deleted']:
                tech_publisher_id = node['id']
                break

        if tech_publisher_id or not response_data['data']['techPublishers']['pageInfo']['hasNextPage']:
            break
        cursor = response_data['data']['techPublishers']['pageInfo']['endCursor']
    
    return tech_publisher_id

# Create a new tech publisher
def create_tech_publisher(publisher_name, api_url, headers):
    mutation = {
        "query": """
        mutation createTechPublisher {
            createTechPublisher(request: {
                name: \"%s\"
                description: \"N/A\"
            }) {
                id
                name
                description
            }
        }
        """ % publisher_name
    }
    response = requests.post(api_url, headers=headers, json=mutation)
    response_data = response.json()
    return response_data['data']['createTechPublisher']['id'] if 'data' in response_data else None

# Create technology mutation
def create_technology(technology_data, tech_publisher_id, api_url, headers):
    mutation = {
        "query": """
        mutation createTechnology {
            createTechnology(request: {
                name: \"%s\"
                description: \"%s\"
                versionSpec: \"%s\"
                techPublishers: {techPublisherId: \"%s\"}
            }) {
                id
                name
                description
            }
        }
        """ % (technology_data['name'], technology_data['description'], technology_data['versionSpec'], tech_publisher_id)
    }
    response = requests.post(api_url, headers=headers, json=mutation)
    return response.json()

# Main function to process the Excel file and create technologies
def main():
    # Configuration
    excel_file_path = 'technologies.xlsx'  # Path to your Excel file
    api_url = 'https://your-graphql-endpoint.com/graphql'  # Replace with your GraphQL endpoint
    headers = {
        'Authorization': 'Bearer YOUR_ACCESS_TOKEN',  # Replace with your OAuth token
        'Content-Type': 'application/json'
    }

    # Load the Excel data
    technologies = load_excel(excel_file_path)

    # Process each technology and create it using the GraphQL API
    for technology in technologies:
        tech_publisher_id = get_tech_publisher_id(technology['publisherName'], api_url, headers)
        if not tech_publisher_id:
            print(f"Tech publisher not found for: {technology['publisherName']}, creating new tech publisher.")
            tech_publisher_id = create_tech_publisher(technology['publisherName'], api_url, headers)
        
        if tech_publisher_id:
            response = create_technology(technology, tech_publisher_id, api_url, headers)
            print(response)
        else:
            print(f"Failed to create tech publisher for: {technology['publisherName']}")

if __name__ == "__main__":
    main()
