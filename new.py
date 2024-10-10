from flask import Flask, request, jsonify, redirect
from ariadne import QueryType, MutationType, make_executable_schema, graphql_sync, load_schema_from_path
from ariadne.constants import PLAYGROUND_HTML
import base64
import hashlib
import os
import requests

# OAuth and GraphQL details
client_id = 'your_client_id'
authorization_url = 'your_authorization_url'
token_url = 'your_token_url'
scope = 'your_scope'
redirect_uri = 'https://localhost:8010/callback'  # The redirect endpoint in Flask
client_secret = 'your_client_secret'  # if applicable

# Flask app setup
app = Flask(__name__)

# PKCE variables
code_verifier = ''
code_challenge = ''

# Step 1: Generate PKCE code verifier and challenge
def generate_code_verifier():
    return base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8').rstrip('=')

def generate_code_challenge(verifier):
    code_challenge_digest = hashlib.sha256(verifier.encode('utf-8')).digest()
    return base64.urlsafe_b64encode(code_challenge_digest).decode('utf-8').rstrip('=')

@app.route('/')
def index():
    global code_verifier, code_challenge
    # Generate code verifier and challenge
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)

    # Redirect to authorization URL
    auth_url = f"{authorization_url}?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope={scope}&code_challenge={code_challenge}&code_challenge_method=S256"
    return redirect(auth_url)

@app.route('/callback')
def callback():
    global code_verifier
    # Get the authorization code
    authorization_code = request.args.get('code')

    if not authorization_code:
        return "Authorization failed: no code returned", 400

    # Exchange authorization code for access token using PKCE
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

# GraphQL API using Ariadne
query = QueryType()
mutation = MutationType()

# Define the hello query
@query.field("hello")
def resolve_hello(_, info, name="stranger"):
    return f"Hello, {name}!"

# Define the createItem mutation
@mutation.field("createItem")
def resolve_create_item(_, info, name):
    # Add logic to handle item creation (e.g., saving to a database)
    return {
        "success": True,
        "message": f"Item '{name}' created successfully!"
    }

# Load the GraphQL schema from string (could be from a `.graphql` file)
type_defs = """
    type Query {
        hello(name: String): String
    }

    type Mutation {
        createItem(name: String!): CreateItemResponse
    }

    type CreateItemResponse {
        success: Boolean!
        message: String!
    }
"""

# Create the executable schema
schema = make_executable_schema(type_defs, query, mutation)

# Set up GraphQL endpoint
@app.route("/graphql", methods=["GET"])
def graphql_playground():
    return PLAYGROUND_HTML, 200

@app.route("/graphql", methods=["POST"])
def graphql_server():
    data = request.get_json()
    success, result = graphql_sync(
        schema,
        data,
        context_value=request,
        debug=app.debug
    )
    status_code = 200 if success else 400
    return jsonify(result), status_code

# Step 5: Run the server on HTTPS (localhost:8010)
if __name__ == '__main__':
    app.run(ssl_context=('cert.pem', 'key.pem'), port=8010)
