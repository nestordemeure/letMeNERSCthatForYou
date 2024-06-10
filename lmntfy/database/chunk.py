class Chunk:
    """Represents a piece of text and its source in the documentation."""
    def __init__(self, url:str, content:str, is_markdown:bool=False):
        """
        url (str): the url to the page or section containing the chunk (might be larger than the chunk)
        content (str): the actual text of the chunk
        is_markdown (bool, default to False): wehther the text is markdown formated
        """
        self.url = url
        self.content = content.strip()
        self.is_markdown = is_markdown

    def __str__(self):
        """turns a chunk into a string representation suitable for usage in a prompt"""
        return f"URL: {self.url}\n\n{self.content}"

    def to_markdown(self):
        """turns a chunk into a markdown representation suitable for usage in a prompt"""
        return (f"Source URL: <{self.url}>\n"
                "Extract:\n"
                "````md\n"
                f"{self.content}\n"
                "````\n\n")

    def to_xml(self):
        """turns a chunk into an XML representation suitable for usage in a prompt"""
        return ("<resource>\n"
                f"<url>{self.url}</url>\n"
                "<text>\n"
                f"{self.content}\n"
                "</text>\n"
                "</resource>")

    def __eq__(self, other):
        if not isinstance(other, Chunk):
            return False
        return (self.url == other.url) and (self.content == other.content)

    def __hash__(self):
        return hash((self.url, self.content))

    def to_dict(self):
        return {
            'url': self.url,
            'content': self.content,
            'is_markdown': self.is_markdown
        }

    @staticmethod
    def from_dict(data):
        return Chunk(data['url'], data['content'], data['is_markdown'])