import requests
import random
import string
import base64
import hashlib
from urllib.parse import urlencode, urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, HTTPServer

# Step 1: Create Code Verifier and Code Challenge
def generate_code_verifier(length=43):
    return ''.join(random.choices(string.ascii_letters + string.digits + '-._~', k=length))

def generate_code_challenge(verifier):
    digest = hashlib.sha256(verifier.encode('utf-8')).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b'=').decode('utf-8')

# Step 2: Set up OAuth URLs and Parameters
client_id = 'YOUR_CLIENT_ID'
redirect_uri = 'http://localhost:8010'
authorization_endpoint = 'https://your-auth-server.com/authorize'
token_endpoint = 'https://your-auth-server.com/token'

verifier = generate_code_verifier()
challenge = generate_code_challenge(verifier)

auth_params = {
    'response_type': 'code',
    'client_id': client_id,
    'redirect_uri': redirect_uri,
    'scope': 'openid profile',  # Add relevant scopes for your GraphQL API
    'code_challenge': challenge,
    'code_challenge_method': 'S256'
}

# Step 3: Start the Authorization Flow
print("Visit the following URL to authorize:")
print(f"{authorization_endpoint}?{urlencode(auth_params)}")

# Step 4: Set up HTTP server to receive the authorization code
class AuthorizationHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        if 'code' in query_params:
            authorization_code = query_params['code'][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Authorization successful! You can close this window.</h1></body></html>")

            # Exchange Authorization Code for Access Token
            token_params = {
                'grant_type': 'authorization_code',
                'code': authorization_code,
                'redirect_uri': redirect_uri,
                'client_id': client_id,
                'code_verifier': verifier
            }

            response = requests.post(token_endpoint, data=token_params)
            response_data = response.json()

            if 'access_token' in response_data:
                access_token = response_data['access_token']
                print("Access Token obtained successfully")
                # Step 5: Use the Access Token to Access the GraphQL API
                graphql_endpoint = "https://your-graphql-endpoint.com/graphql"
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }

                query = """
                query {
                    someData {
                        field1
                        field2
                    }
                }
                """

                graphql_response = requests.post(graphql_endpoint, headers=headers, json={"query": query})
                print(graphql_response.json())
            else:
                print("Failed to get Access Token")
                print(response_data)

        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Authorization failed or canceled.</h1></body></html>")

server_address = ('', 8010)
httpd = HTTPServer(server_address, AuthorizationHandler)
httpd.socket = ssl.wrap_socket(httpd.socket, certfile='./server.pem', server_side=True)
print('Starting HTTPS server on https://localhost:8010...')
httpd.serve_forever()
