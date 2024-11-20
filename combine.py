import glob
import re

# Initialize data structures
objects = {}
connections = set()
leftover_connections = set()

# Define a function to parse each analysis file
def parse_analysis(file_content):
    obj_section = False
    conn_section = False
    leftover_section = False
    current_boundary = ''
    current_trust_level = ''
    
    for line in file_content.splitlines():
        # Check for section headers
        if line.startswith('- **Objects and Boundaries:**'):
            obj_section = True
            conn_section = False
            leftover_section = False
            continue
        elif line.startswith('- **Connections:**'):
            obj_section = False
            conn_section = True
            leftover_section = False
            continue
        elif line.startswith('- **Leftover Connections:**'):
            obj_section = False
            conn_section = False
            leftover_section = True
            continue
        
        # Parse objects
        if obj_section:
            boundary_match = re.match(r'\s*-\s*(.+):', line)
            if boundary_match:
                current_boundary = boundary_match.group(1).strip()
                continue
            trust_level_match = re.match(r'\s*- (.+):', line)
            if trust_level_match:
                current_trust_level = trust_level_match.group(1).strip()
                continue
            object_match = re.match(r'\s*- (.+)', line)
            if object_match:
                obj_name = object_match.group(1).strip()
                objects[obj_name] = {
                    'boundary': current_boundary,
                    'trust_level': current_trust_level
                }
        # Parse connections
        elif conn_section:
            conn_match = re.match(r'\d+\.\s*(.+)', line)
            if conn_match:
                connection = conn_match.group(1).strip()
                connections.add(connection)
        # Parse leftover connections
        elif leftover_section:
            if line.strip():
                leftover_connections.add(line.strip())

# Process each analysis file
analysis_files = glob.glob('Analysis_Section*.txt')
for file_name in analysis_files:
    with open(file_name, 'r') as file:
        content = file.read()
        parse_analysis(content)

# Now, objects, connections, and leftover_connections contain combined data
# You can now format and write this data into your final document