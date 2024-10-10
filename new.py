import pandas as pd
import requests
from requests_oauthlib import OAuth2Session

# OAuth Configuration
client_id = 'your_client_id'
authorization_base_url = 'https://your-oauth-provider.com/auth'
token_url = 'https://your-oauth-provider.com/token'
redirect_uri = 'https://your-redirect-uri.com/callback'
scope = ['your_scope']  # e.g., ['read', 'write']
graphql_url = 'https://your-graphql-api.com/graphql'

# Step 1: Redirect user to OAuth provider for authorization
def get_authorization_code():
    oauth = OAuth2Session(client_id, scope=scope, redirect_uri=redirect_uri)
    authorization_url, state = oauth.authorization_url(authorization_base_url)

    print('Please go to this URL and authorize access:', authorization_url)
    # The user will get redirected to a URL with an authorization code after authorization
    # Paste that URL here:
    redirect_response = input('Paste the full redirect URL here: ')

    # Extract the authorization code from the redirect URL
    token = oauth.fetch_token(token_url, authorization_response=redirect_response)
    return token['access_token']

# Test GraphQL connection with a simple query
def test_graphql_connection(token):
    test_query = """
    query {
        __schema {
            queryType {
                name
            }
        }
    }
    """
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    response = requests.post(graphql_url, json={'query': test_query}, headers=headers)
    return response.status_code, response.json()

# Read CSV file
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Create GraphQL mutation based on CSV row
def create_graphql_mutation(row):
    mutation = """
    mutation {
        createRecord(input: {
            field1: "%s",
            field2: "%s",
            field3: "%s"
        }) {
            id
        }
    }
    """ % (row['field1'], row['field2'], row['field3'])
    return mutation

# Send mutation request to GraphQL API
def send_graphql_request(mutation, token):
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    response = requests.post(graphql_url, json={'query': mutation}, headers=headers)
    return response.json()

# Main function
def main():
    token = get_authorization_code()

    # Test connection
    status_code, response = test_graphql_connection(token)
    if status_code == 200:
        print("Connection successful:", response)
    else:
        print("Connection failed:", response)
        return

    # Proceed with mutation if the connection is successful
    csv_data = read_csv('your_file.csv')
    for _, row in csv_data.iterrows():
        mutation = create_graphql_mutation(row)
        response = send_graphql_request(mutation, token)
        print(response)

if __name__ == '__main__':
    main()
