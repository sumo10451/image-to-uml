import requests
from requests_oauthlib import OAuth2Session
from graphqlclient import GraphQLClient

# Replace these values with your actual configuration
client_id = 'YOUR_CLIENT_ID'
authorization_base_url = 'https://example.com/oauth/authorize'  # Replace with actual authorization URL
token_url = 'https://example.com/oauth/token'  # Replace with actual token URL
redirect_uri = 'https://localhost:8501/callback'

# Step 1: Start OAuth Session
oauth = OAuth2Session(client_id, redirect_uri=redirect_uri)

# Step 2: Redirect user for authorization
authorization_url, state = oauth.authorization_url(authorization_base_url)
print('Please go to %s and authorize access.' % authorization_url)

# User will receive a code from the redirect, use that code here
redirect_response = input('Paste the full redirect URL here: ')

# Step 3: Fetch the token using the authorization code (without client secret)
token = oauth.fetch_token(token_url, authorization_response=redirect_response)

# Step 4: Access GraphQL API using token
client = GraphQLClient('https://example.com/graphql')  # Replace with your GraphQL endpoint
client.inject_token(token['access_token'], 'Bearer')

# Define your GraphQL query or mutation
query = """
{
    sampleQuery {
        field1
        field2
    }
}
"""

# Execute the query
try:
    result = client.execute(query)
    print(result)
except Exception as e:
    print(f"Error occurred: {e}")
