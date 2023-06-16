from pathlib import Path
from urllib.parse import quote

def path2url(file_path:Path, documentation_folder_path:Path):
    """
    takes a file path and the documentation folder containing the file
    outputs the corresponding url
    """
    # remove the documentation folder from the path
    relative_path = file_path.relative_to(documentation_folder_path)
    # Construct the new URL
    url = "https://docs.nersc.gov/" + relative_path
    # Replace an optional '/index.md' with '/'
    url = url.replace("/index.md", "/")
    # Replace the final '.md' with '/'
    url = url.replace(".md", "/")
    # Convert spaces into URL valid format
    url = quote(url, safe='/:')
    return url

class Chunk:
    def __init__(self, source, url, content):
        self.source = Path(source)
        self.url = url
        self.content = content

    def __str__(self):
        """turns a chunk into a string representation suitable for usage in a prompt"""
        return f"URL: {self.url}\n\n{self.content}"

    def to_dict(self):
        return {
            'source': str(self.source),
            'url': self.url,
            'content': self.content
        }

    @staticmethod
    def from_dict(data):
        return Chunk(data['source'], data['url'], data['content'])