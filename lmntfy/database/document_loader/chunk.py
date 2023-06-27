from pathlib import Path
from urllib.parse import quote
import re

def path2url(file_path:Path, documentation_folder_path:Path):
    """
    takes a file path and the documentation folder containing the file
    outputs the corresponding url
    """
    # remove the documentation folder from the path
    relative_path = file_path.relative_to(documentation_folder_path)
    # Construct the new URL
    url = "https://docs.nersc.gov/" + str(relative_path)
    # Replace an optional '/index.md' with '/'
    url = url.replace("/index.md", "/")
    # Replace the final '.md' with '/'
    url = url.replace(".md", "/")
    # Convert spaces into URL valid format
    url = quote(url, safe='/:#')
    return url

def header2url(url:str, header:str):
    """
    takes a url and the header of a markdown section inside that file
    converts it into a url to a particular part of that page
    """
    # extract the heading
    lines = header.splitlines()
    heading = lines[0] if header.startswith('#') else ""
    # converts the heading into a proper url format    
    # Convert to lowercase
    heading = heading.lower()
    # Remove all non-alphanumeric characters
    heading = re.sub('[^a-z0-9- ]', '', heading)
    # strip side spaces
    heading = heading.strip()
    # Replace spaces with dashes
    heading = heading.replace(' ', '-')
    # Add back the hash sign at the start for URL fragment identifier
    heading = '#' + heading
    # merges it with the url
    return url + heading

class Chunk:
    def __init__(self, url, content):
        self.url = url
        self.content = content.strip()

    def __str__(self):
        """turns a chunk into a string representation suitable for usage in a prompt"""
        return f"URL: {self.url}\n\n{self.content}"

    def to_dict(self):
        return {
            'url': self.url,
            'content': self.content
        }

    @staticmethod
    def from_dict(data):
        return Chunk(data['url'], data['content'])