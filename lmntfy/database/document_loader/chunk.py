from pathlib import Path

class Chunk:
    def __init__(self, source=None, content=None):
        self.source = Path(source) if source else None
        self.content = content

    def to_dict(self):
        return {
            'source': str(self.source),
            'content': self.content
        }

    @staticmethod
    def from_dict(data):
        source = Path(data['source'])
        content = data['content']
        return Chunk(source, content)