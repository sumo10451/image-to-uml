import base64
import hashlib
import os
import requests
import pandas as pd

# OAuth and GraphQL details
client_id = 'your_client_id'
authorization_url = 'your_authorization_url'
token_url = 'your_token_url'
scope = 'your_scope'
redirect_uri = 'https://localhost:8010'
graphql_url = 'your_graphql_url'
csv_file = 'path_to_your_csv_file.csv'
client_secret = 'your_client_secret'  # if applicable

# Step 1: Generate a code verifier and code challenge
def generate_code_verifier():
    return base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8').rstrip('=')

def generate_code_challenge(verifier):
    code_challenge_digest = hashlib.sha256(verifier.encode('utf-8')).digest()
    return base64.urlsafe_b64encode(code_challenge_digest).decode('utf-8').rstrip('=')

# Step 2: Get the authorization code with PKCE
code_verifier = generate_code_verifier()
code_challenge = generate_code_challenge(code_verifier)

# Step 3: Authorize the app
auth_response = requests.get(
    authorization_url,
    params={
        'response_type': 'code',
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'scope': scope,
        'code_challenge': code_challenge,
        'code_challenge_method': 'S256'
    }
)
print("Go to the following URL to authorize the application:")
print(auth_response.url)

# Step 4: After the user authorizes, they will be redirected with an authorization code
authorization_code = input("Enter the authorization code from the redirected URL: ")

# Step 5: Exchange authorization code for access token
token_response = requests.post(
    token_url,
    data={
        'grant_type': 'authorization_code',
        'code': authorization_code,
        'redirect_uri': redirect_uri,
        'client_id': client_id,
        'code_verifier': code_verifier,
        'client_secret': client_secret  # if required by your OAuth provider
    }
)
token_data = token_response.json()
access_token = token_data.get('access_token')

if not access_token:
    raise Exception(f"Failed to get access token: {token_data}")

print(f"Access token: {access_token}")

# Step 6: Read CSV file and prepare GraphQL mutation
df = pd.read_csv(csv_file)

mutation_template = """
mutation($input: YourInputType!) {
    createItem(input: $input) {
        id
        status
    }
}
"""

headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
}

# Step 7: Send mutations to GraphQL API
for index, row in df.iterrows():
    # Prepare the input data for the mutation based on CSV columns
    input_data = {
        "field1": row['column1'],
        "field2": row['column2'],
        # Add more fields as per your GraphQL mutation requirements
    }

    response = requests.post(
        graphql_url,
        json={
            'query': mutation_template,
            'variables': {'input': input_data}
        },
        headers=headers
    )

    result = response.json()
    print(result)

    if 'errors' in result:
        print(f"Error for row {index}: {result['errors']}")
