import base64
import hashlib
import os
import requests
from flask import Flask, request, redirect

# OAuth and GraphQL details
client_id = 'your_client_id'
authorization_url = 'your_authorization_url'
token_url = 'your_token_url'
scope = 'your_scope'
redirect_uri = 'https://localhost:8010/callback'  # The redirect endpoint in Flask
graphql_url = 'your_graphql_url'
csv_file = 'path_to_your_csv_file.csv'
client_secret = 'your_client_secret'  # if applicable

# Flask app setup
app = Flask(__name__)

# Generate PKCE code verifier and challenge
def generate_code_verifier():
    return base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8').rstrip('=')

def generate_code_challenge(verifier):
    code_challenge_digest = hashlib.sha256(verifier.encode('utf-8')).digest()
    return base64.urlsafe_b64encode(code_challenge_digest).decode('utf-8').rstrip('=')

# PKCE variables (these will be generated dynamically)
code_verifier = ''
code_challenge = ''

@app.route('/')
def index():
    global code_verifier, code_challenge
    # Step 1: Generate code verifier and code challenge for PKCE
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)

    # Step 2: Redirect user to the authorization URL
    auth_url = f"{authorization_url}?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope={scope}&code_challenge={code_challenge}&code_challenge_method=S256"
    return redirect(auth_url)

@app.route('/callback')
def callback():
    global code_verifier
    # Step 3: Get the authorization code from the callback URL
    authorization_code = request.args.get('code')

    if not authorization_code:
        return "Authorization failed: no code returned", 400

    # Step 4: Exchange authorization code for access token using PKCE
    token_response = requests.post(
        token_url,
        data={
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'redirect_uri': redirect_uri,
            'client_id': client_id,
            'code_verifier': code_verifier,
            'client_secret': client_secret  # if required
        }
    )

    token_data = token_response.json()
    access_token = token_data.get('access_token')

    if not access_token:
        return f"Failed to get access token: {token_data}", 400

    return f"Access token: {access_token}"

# Step 5: Run the server on HTTPS (localhost:8010)
if __name__ == '__main__':
    app.run(ssl_context=('cert.pem', 'key.pem'), port=8010)
