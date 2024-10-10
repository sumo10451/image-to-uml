import pandas as pd
import requests
from requests_oauthlib import OAuth2Session

# OAuth Configuration
client_id = 'your_client_id'
client_secret = 'your_client_secret'
token_url = 'https://your-oauth-provider.com/token'
graphql_url = 'https://your-graphql-api.com/graphql'

# Fetch OAuth token
def get_oauth_token():
    oauth = OAuth2Session(client_id)
    token = oauth.fetch_token(token_url=token_url, client_id=client_id, client_secret=client_secret)
    return token['access_token']

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
    token = get_oauth_token()
    csv_data = read_csv('your_file.csv')

    for _, row in csv_data.iterrows():
        mutation = create_graphql_mutation(row)
        response = send_graphql_request(mutation, token)
        print(response)

if __name__ == '__main__':
    main()
