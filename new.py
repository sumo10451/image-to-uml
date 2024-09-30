import requests
from requests.auth import HTTPBasicAuth

# Replace with your Azure DevOps organization URL, project name, and personal access token (PAT)
organization_url = "https://dev.azure.com/{organization}"
project = "{project}"
ticket_number = "{work_item_id}"  # Replace with your actual work item (ticket) number
personal_access_token = "your_pat_here"  # Replace with your actual PAT

# API URL to get work item details
url = f"{organization_url}/{project}/_apis/wit/workitems/{ticket_number}?api-version=7.0"

# Make the request
response = requests.get(url, auth=HTTPBasicAuth('', personal_access_token))

# Check if the request was successful
if response.status_code == 200:
    # Print the work item details
    work_item = response.json()
    print("Ticket Information:")
    print(f"ID: {work_item['id']}")
    print(f"Title: {work_item['fields']['System.Title']}")
    print(f"State: {work_item['fields']['System.State']}")
    print(f"Assigned To: {work_item['fields'].get('System.AssignedTo', {}).get('displayName', 'Unassigned')}")
else:
    print(f"Failed to get work item. Status code: {response.status_code}, Message: {response.text}")