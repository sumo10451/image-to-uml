import zipfile
import xml.etree.ElementTree as ET

# For .vsdx files, which are essentially zip files containing XML documents
with zipfile.ZipFile('diagram.vsdx', 'r') as visio_zip:
    with visio_zip.open('visio/pages/page1.xml') as page_xml:
        tree = ET.parse(page_xml)
        root = tree.getroot()

        ns = {'visio': 'http://schemas.microsoft.com/office/visio/2012/main'}

        for connect in root.findall('.//visio:Connect', ns):
            from_id = connect.attrib.get('FromSheet')
            to_id = connect.attrib.get('ToSheet')
            print(f"Connection from shape ID {from_id} to shape ID {to_id}")