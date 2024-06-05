import re
from .text_splitter import text_splitter
from ..chunk import Chunk
from .path_to_url import addHeader2url
from typing import Callable, List

#----------------------------------------------------------------------------------------
# MARKDOWN PARSING

class Markdown:
    """
    Tree representation for a markdown file
    """
    def __init__(self, header: str, level: int, headings: list=[]):
        self.header = header 
        self.level = level
        self.headings = headings
        self.nb_tokens = None

    @staticmethod
    def load(markdown_text: str):
        result = Markdown(header="", level=0, headings=[])
        in_code_block = False
        for line in markdown_text.splitlines():
            # flip the state if this is a codeblock
            if line.startswith("```"):
                in_code_block = not in_code_block
            # inserts either text or a heading
            match = re.match(r'^#+', line)
            if match and not in_code_block:
                level = len(match.group())
                result.insert_heading(text=line, level=level)
            else:
                result.insert_text(line)
        # pop the upper level if it is empty
        if (len(result.header) == 0) and (len(result.headings) == 1):
            result = result.headings[0]
        return result
    
    def insert_text(self, text):
        """insert text at the end of the header of the latest, deepest, heading to data"""
        if len(self.headings) == 0:
            # we have reached the bottom
            self.header += '\n' + text
        else:
            self.headings[-1].insert_text(text)

    def insert_heading(self, text, level):
        """insert a new heading whereaver is possible"""
        if (len(self.headings) == 0) or (level <= self.headings[-1].level):
            heading = Markdown(text, level, headings=[])
            self.headings.append(heading)
        else:
            self.headings[-1].insert_heading(text, level)

    def to_string(self):
        result = self.header + '\n'
        for heading in self.headings:
            result += '\n' + heading.to_string()
        return result

    def count_tokens(self, token_counter):
        """memoized token counting function"""
        if self.nb_tokens is None:
            self.nb_tokens = token_counter(self.header) + sum((heading.count_tokens(token_counter) for heading in self.headings), 0)
        return self.nb_tokens

    def to_chunks(self, url, token_counter, max_tokens):
        # computes the url to the current heading
        local_url = addHeader2url(url, self.header)
        # computes the current chunks
        if (self.count_tokens(token_counter) < max_tokens) and (len(self.headings) > 0):
            # small enough to fit
            return [Chunk(url=local_url, content=self.to_string())]
        else:
            # split along headings
            # only turns header into chunks if it has more than a title
            header = self.header.strip()
            if ('\n' in header):
                result = text_splitter(local_url, header, token_counter, max_tokens)
            else:
                result = []
            # include all subheadings
            for heading in self.headings:
                result.extend(heading.to_chunks(url, token_counter, max_tokens))
            return result

#----------------------------------------------------------------------------------------
# PUBLIC FUNCTION

def markdown_splitter(url:str, markdown:str, token_counter:Callable[[str], int], max_tokens:int) -> List[Chunk]:
    """
    Splits a given markdown file into chunks based on a maximum token limit.

    Args:
        url (str): The URL where the markdown file can be found.
        markdown (str): The input markdown content to be split.
        token_counter (Callable[[str], int]): A function that returns the number of tokens in a given string.
        max_tokens (int): The maximum number of tokens allowed in each chunk.

    Returns:
        List[Chunk]: A list of chunks, each having a URL derived from he given one (taking headers into account) and containing no more than the specified maximum number of tokens.
                     All having `is_markdown=True`.

    The function parses the markdown content into a tree representation based on its headings. It recursively splits the content until each chunk is small enough to fit within the maximum number of tokens allowed.
    """
    # shortcut if the text is short enough to be returned uncut
    if token_counter(markdown) < max_tokens:
        return [Chunk(url=url, content=markdown, is_markdown=True)]
    # parses the text into a tree representation
    ast = Markdown.load(markdown)
    # turn it into a list of chunks of appropriate size
    chunks = ast.to_chunks(url, token_counter, max_tokens)
    # labels all chunks as markdown
    for chunk in chunks:
        chunk.is_markdown = True
    return chunks
