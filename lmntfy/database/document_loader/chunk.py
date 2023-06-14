from pathlib import Path
from urllib.parse import quote

class Chunk:
    def __init__(self, source, content):
        self.source = Path(source)
        self.content = content
        self._url = None # used to memoize the url

    @property
    def url(self) -> str:
        """url to the source"""
        if self._url is None:
            # Partition the path at the first occurrence of '/docs/' and take the part after it
            _, _, post_docs_path = str(self.source).partition('/docs/')
            # Construct the new URL
            url = "https://docs.nersc.gov/" + post_docs_path
            # Replace an optional '/index.md' with '/'
            url = url.replace("/index.md", "/")
            # Replace the final '.md' with '/'
            url = url.replace(".md", "/")
            # Convert spaces into URL valid format
            url = quote(url, safe='/:')
            self._url = url
        return self._url

    def __str__(self):
        """turns a chunk into a string representation suitable for usage in a prompt"""
        return f"URL: {self.url}\n\n{self.content}"

    def to_dict(self):
        return {
            'source': str(self.source),
            'content': self.content
        }

    @staticmethod
    def from_dict(data):
        return Chunk(data['source'], data['content'])