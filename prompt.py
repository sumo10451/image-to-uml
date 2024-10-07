def create_threat_model_prompt(app_type, authentication, internet_facing, sensitive_data, app_input):
    prompt = f"""
Act as a cybersecurity expert with more than 20 years of experience in using the STRIDE threat modeling methodology to produce detailed threat scenarios. Your task is to analyze the provided code summary, README content, and application description to produce specific threats for the application.

Pay special attention to the README content as it often provides valuable context about the project's purpose, architecture, and potential security considerations.

For each of the STRIDE categories (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, and Elevation of Privilege), provide a threat scenario in the following format:

1. **Threat Title**: A brief description of the threat.
   **Category**: {STRIDE category}
   **Description**: A specific explanation of how the threat could occur.
   **Justification**: If no specific mitigation exists, mention it here.
   **Possible Mitigation**: How this threat could be prevented or mitigated.
   **SDL Phase**: In which phase of the Secure Development Lifecycle (e.g., design, implementation, testing) this should be addressed.

APPLICATION TYPE: {app_type}
AUTHENTICATION METHODS: {authentication}
INTERNET FACING: {internet_facing}
SENSITIVE DATA: {sensitive_data}
CODE SUMMARY, README CONTENT, AND APPLICATION DESCRIPTION:
{app_input}

Example output format:
  
1. **Threat Title**: Unauthorized access to MySQL DB
   **Category**: Elevation of Privilege
   **Description**: An adversary can gain unauthorized access to Azure MySQL DB instances due to weak network security configuration.
   **Justification**: No specific mitigation provided.
   **Possible Mitigation**: Restrict access by configuring server-level firewall rules to permit only trusted IP addresses.
   **SDL Phase**: Implementation

    """
    return prompt
